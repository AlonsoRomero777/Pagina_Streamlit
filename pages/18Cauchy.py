import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

st.title("Método de Cauchy")
st.markdown("""
<div class="justify">
<p><strong>El Método de Cauchy</strong>, también conocido como el <strong>método del gradiente más pronunciado</strong>, es un algoritmo fundamental en optimización numérica. Su objetivo es encontrar el mínimo de una función real y diferenciable, siguiendo la dirección de descenso más rápido: el gradiente negativo. Se considera una versión básica del descenso de gradiente.</p>

<p>En cada iteración, el método calcula el gradiente de la función en el punto actual y avanza en la dirección opuesta, que es la dirección en la que la función decrece con mayor rapidez. Para decidir cuánto avanzar, se utiliza una <strong>búsqueda unidimensional</strong> (line search), comúnmente con métodos como la <em>búsqueda dorada</em> o la <em>búsqueda por intervalos</em>, para encontrar la longitud de paso (α) óptima que minimiza la función a lo largo de esa dirección.</p>

<p>Matemáticamente, la actualización de cada iteración se expresa como:</p>
<center>
x<sub>k+1</sub> = x<sub>k</sub> - α<sub>k</sub> ∇f(x<sub>k</sub>)
</center>

<p>Donde:</p>
<ul>
  <li><strong>∇f(x<sub>k</sub>)</strong> es el gradiente de la función en el punto x<sub>k</sub>.</li>
  <li><strong>α<sub>k</sub></strong> es el paso óptimo calculado mediante búsqueda unidimensional.</li>
</ul>

<p><strong>Ventajas: 👍</strong></p>
<ul>
  <li>Fácil de implementar.</li>
  <li>No requiere cálculo de segundas derivadas (Hessiana).</li>
  <li>Puede aplicarse a funciones de muchas variables.</li>
</ul>

<p><strong>Desventajas: 👎</strong></p>
<ul>
  <li>Convergencia lenta cerca del óptimo.</li>
  <li>Sensible al valor inicial y a la escala de la función.</li>
  <li>No es eficiente en funciones mal condicionadas o con valles estrechos (como Rosenbrock).</li>
</ul>

<p>En la práctica, el Método de Cauchy suele utilizarse como base o referencia para otros métodos más sofisticados, como Newton-Raphson, Quasi-Newton (BFGS) o métodos conjugados.</p>
</div>
""", unsafe_allow_html=True)


def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def rosenbrock(x):
    return sum([100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1)])

def ackley(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum1 = sum([xi**2 for xi in x])
    sum2 = sum([np.cos(c * xi) for xi in x])
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

def sphere(x):
    return sum([xi**2 for xi in x])

def beale(x):
    x1, x2 = x[0], x[1]
    return ((1.5 - x1 + x1 * x2)**2 + 
            (2.25 - x1 + x1 * x2**2)**2 + 
            (2.625 - x1 + x1 * x2**3)**2)

def booth(x):
    x1, x2 = x[0], x[1]
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

def himmelblau(x):
    x1, x2 = x[0], x[1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def mccormick(x):
    x1, x2 = x[0], x[1]
    return np.sin(x1 + x2) + (x1 - x2)**2 - 1.5 * x1 + 2.5 * x2 + 1

funciones = {
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock,
    "Ackley": ackley,
    "Sphere": sphere,
    "Beale": beale,
    "Booth": booth,
    "Himmelblau": himmelblau,
    "McCormick": mccormick,
}

def gradiente(function, x, h=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (function(x_plus) - function(x_minus)) / (2 * h)
    return grad

def regla_eliminacion(x1, x2, fx1, fx2, a, b):
    if fx1 > fx2:
        return x1, b
    if fx1 < fx2:
        return a, x2
    return x1, x2

def w_to_x(w, a, b):
    return w * (b - a) + a

def busquedad_orada(funcion, epsilon, a=0.0, b=1.0):
    PHI = (1 + np.sqrt(5)) / 2 - 1
    aw, bw = 0, 1
    while (bw - aw) > epsilon:
        w2 = aw + PHI * (bw - aw)
        w1 = bw - PHI * (bw - aw)
        aw, bw = regla_eliminacion(w1, w2, funcion(w_to_x(w1, a, b)), funcion(w_to_x(w2, a, b)), aw, bw)
    return (w_to_x(aw, a, b) + w_to_x(bw, a, b)) / 2

def cauchy(function, x0, epsilon1, epsilon2, M):
    xk = x0
    path = [xk.copy()]
    for k in range(M):
        grad = gradiente(function, xk)
        if np.linalg.norm(grad) < epsilon1:
            break
        alpha_f = lambda alpha: function(xk - alpha * grad)
        alpha = busquedad_orada(alpha_f, epsilon2)
        xk1 = xk - alpha * grad
        path.append(xk1.copy())
        if np.linalg.norm(xk1 - xk) / (np.linalg.norm(xk) + 1e-10) < epsilon2:
            break
        xk = xk1
    return xk, function(xk), path

nombre_funcion = st.selectbox("Selecciona una función objetivo", list(funciones.keys()))
funcion = funciones[nombre_funcion]

epsilon1 = st.slider("Tolerancia del gradiente (ε₁)", 0.0001, 0.01, 0.001, step=0.0001)
epsilon2 = st.slider("Tolerancia relativa (ε₂)", 0.0001, 0.01, 0.001, step=0.0001)
max_iter = st.slider("Máximo de iteraciones", 50, 1000, 200, step=50)

np.random.seed(42)
x0 = np.random.uniform(-5, 5, size=2)

sol, val, path = cauchy(funcion, x0, epsilon1, epsilon2, max_iter)

st.markdown(f"**Mejor solución encontrada:** `{sol}`")
st.markdown(f"**Valor mínimo estimado:** `{val:.6f}`")


path = np.array(path)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(path[:, 0], path[:, 1], marker='o', markersize=2, linestyle='-', alpha=0.7)
ax.scatter(path[0, 0], path[0, 1], color='green', label='Inicio', s=80)
ax.scatter(path[-1, 0], path[-1, 1], color='red', label='Fin', s=80)

if nombre_funcion == "Rosenbrock":
    ax.scatter(1, 1, color='gold', marker='*', s=200, label='Óptimo teórico')
elif nombre_funcion == "Rastrigin":
    ax.scatter(0, 0, color='gold', marker='*', s=200, label='Óptimo teórico')

ax.set_title(f"Recorrido con Método de Cauchy en {nombre_funcion}")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.grid()
ax.legend()
st.pyplot(fig)
