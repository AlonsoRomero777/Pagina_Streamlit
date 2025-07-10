import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.markdown("""
<style>
.justify { text-align: justify; }
</style>
""", unsafe_allow_html=True)

st.title("Algoritmo de Optimización: Hill Climbing")

st.markdown("""
<div class="justify">
<p><strong>Hill Climbing</strong> es un algoritmo de optimización local que parte de una solución inicial y se mueve paso a paso hacia una mejor solución dentro de su vecindario. En cada iteración, se evalúa una nueva solución vecina y se acepta si mejora el valor de la función objetivo.</p>

<p>Este método es eficiente para encontrar óptimos locales, especialmente cuando el espacio de búsqueda es suave y tiene pocas irregularidades. Sin embargo, una de sus principales limitaciones es que puede quedarse atrapado en un óptimo local si no existen soluciones vecinas mejores.</p>

<p>Existen varias variantes del algoritmo:</p>
<ul>
<li><strong>Hill Climbing Simple:</strong> Evalúa una vecindad aleatoria y acepta solo si mejora la solución.</li>
<li><strong>Steepest Ascent Hill Climbing:</strong> Evalúa todos los vecinos y selecciona el mejor.</li>
<li><strong>Stochastic Hill Climbing:</strong> Selecciona un vecino aleatorio que sea mejor que el actual.</li>
<li><strong>Random-Restart Hill Climbing:</strong> Ejecuta múltiples veces el algoritmo desde diferentes puntos de inicio para evitar óptimos locales.</li>
</ul>

<p>Este algoritmo es útil en problemas donde el espacio de búsqueda es grande pero se puede evaluar rápidamente, como problemas de diseño, afinación de hiperparámetros o funciones de una sola variable.</p>

<p>Aunque es sencillo de implementar, Hill Climbing puede beneficiarse de estrategias complementarias, como recocido simulado o algoritmos genéticos, para mejorar su capacidad de escapar de mínimos locales.</p>
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

nombre_funcion = st.selectbox("Selecciona una función objetivo", list(funciones.keys()))
funcion = funciones[nombre_funcion]

max_iter = st.slider("Número de iteraciones", 100, 5000, 1000, step=100)
sigma = st.slider("Tamaño del paso (sigma)", 0.01, 1.0, 0.3)


def random_generation(x, sigma=0.1):
    return x + np.random.normal(0, sigma, size=len(x))


def hill_climbing(f, x0, max_iter=1000, sigma=0.1):
    x_current = np.copy(x0)
    f_current = f(x_current)
    history = [f_current]
    path = [x_current.copy()]
    
    for iteration in range(max_iter):
        x_k_plus_1 = random_generation(x_current, sigma)
        f_k_plus_1 = f(x_k_plus_1)
        
        if f_k_plus_1 < f_current:
            x_current = x_k_plus_1
            f_current = f_k_plus_1
        
        history.append(f_current)
        path.append(x_current.copy())
    
    return x_current, f_current, history, path

np.random.seed(42)
x0 = np.random.uniform(-5, 5, size=2)

sol, val, hist, path = hill_climbing(funcion, x0, max_iter=max_iter, sigma=sigma)


st.markdown(f"**Mejor solución encontrada:** {sol}")
st.markdown(f"**Valor mínimo de la función:** {val:.6f}")


path = np.array(path)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(path[:, 0], path[:, 1], marker='o', markersize=2, linestyle='-', alpha=0.7)
ax.scatter(path[0, 0], path[0, 1], color='green', label='Inicio', s=80)
ax.scatter(path[-1, 0], path[-1, 1], color='red', label='Fin', s=80)

if nombre_funcion == "Rosenbrock":
    ax.scatter(1, 1, color='gold', marker='*', s=200, label='Óptimo teórico')
elif nombre_funcion == "Rastrigin":
    ax.scatter(0, 0, color='gold', marker='*', s=200, label='Óptimo teórico')

ax.set_title(f'Recorrido en {nombre_funcion}')
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.grid()
ax.legend()
st.pyplot(fig)
