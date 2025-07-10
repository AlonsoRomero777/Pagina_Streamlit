import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.markdown("""
<style>
.justify { text-align: justify; }
</style>
""", unsafe_allow_html=True)

st.title("Algoritmo de Optimización: Recocido Simulado")

st.markdown("""
<div class="justify">
<p><strong>El Recocido Simulado</strong> es un algoritmo de optimización inspirado en el proceso físico de recocido de metales, donde un material es calentado y luego enfriado lentamente para alcanzar una estructura más estable (menor energía interna).</p>

<p>En el contexto de la optimización, el algoritmo busca el mínimo global de una función simulando este proceso térmico. Se permite aceptar soluciones peores temporalmente con cierta probabilidad, para evitar quedarse atrapado en mínimos locales.</p>

<p>El algoritmo comienza con una solución inicial y una temperatura alta. A cada paso, genera una nueva solución candidata en el vecindario de la actual. Si la nueva solución es mejor, se acepta directamente. Si es peor, se acepta con una probabilidad que depende de la diferencia entre ambas soluciones y de la temperatura actual.</p>

<p>A medida que avanza la ejecución, la temperatura disminuye gradualmente siguiendo un esquema de enfriamiento (por ejemplo, lineal, exponencial o logarítmico). Esto reduce la probabilidad de aceptar soluciones peores, afinando la búsqueda hacia un óptimo global.</p>

<p>El Recocido Simulado es útil para problemas complejos, especialmente aquellos con múltiples óptimos locales, como el problema del viajero (TSP), diseño de circuitos, y optimización combinatoria.</p>

<p>Parámetros clave del algoritmo:</p>
<ul>
<li><strong>Temperatura inicial (T<sub>0</sub>):</strong> controla el grado de exploración al inicio.</li>
<li><strong>Enfriamiento:</strong> define cómo se reduce la temperatura (por ejemplo, <code>T = T * α</code>).</li>
<li><strong>Función de vecindario:</strong> determina cómo se generan nuevas soluciones.</li>
<li><strong>Criterio de parada:</strong> puede ser un número fijo de iteraciones, temperatura mínima o estabilidad en la solución.</li>
</ul>

<p>Gracias a su simplicidad y capacidad de escapar de óptimos locales, el Recocido Simulado sigue siendo una técnica valiosa en la optimización moderna.</p>
</div>
""", unsafe_allow_html=True)

def tweak(x, sigma=0.1):
    return x + np.random.normal(0, sigma, size=len(x))

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
w = st.slider("Tamaño del vecindario (w)", 1, 100, 20)
alpha = st.slider("Factor de enfriamiento (α)", 0.80, 0.99, 0.95)
sigma = st.slider("Tamaño del paso (sigma)", 0.01, 1.0, 0.3)

def simulated_annealing(f, x0, max_iter=500, w=20, alpha=0.95, sigma=0.3):
    X = np.copy(x0)
    Best = np.copy(x0)
    T = 1.0
    history = [f(Best)]
    path = [Best.copy()]
    temperatures = [T]
    iteration = 0

    while iteration < max_iter and T > 1e-8:
        for _ in range(w):
            U = tweak(X, sigma)
            if f(U) < f(Best):
                Best = np.copy(U)
                X = np.copy(U)
            else:
                delta = f(U) - f(X)
                if np.exp(-delta / T) >= np.random.uniform(0, 1):
                    X = np.copy(U)
            history.append(f(Best))
            path.append(Best.copy())
        T *= alpha
        temperatures.append(T)
        iteration += 1

    return Best, f(Best), history, path, temperatures

np.random.seed(42)
x0 = np.random.uniform(-5, 5, size=2)

sol, val, hist, path, temps = simulated_annealing(
    funcion, x0, max_iter=max_iter, w=w, alpha=alpha, sigma=sigma
)

st.markdown(f"**Mejor solución encontrada:** `{sol}`")
st.markdown(f"**Valor mínimo estimado:** `{val:.6f}`")

path = np.array(path)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(path[:, 0], path[:, 1], marker='o', markersize=2, linestyle='-', alpha=0.7)
ax.scatter(path[0, 0], path[0, 1], color='green', label='Inicio', s=80)
ax.scatter(path[-1, 0], path[-1, 1], color='red', label='Fin', s=80)

if nombre_funcion == "Rosenbrock":
    ax.scatter(1, 1, color='gold', marker='*', s=200, label='Óptimo teórico')
elif nombre_funcion == "Rastrigin":
    ax.scatter(0, 0, color='gold', marker='*', s=200, label='Óptimo teórico')

ax.set_title(f"Recorrido en {nombre_funcion}")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.grid()
ax.legend()
st.pyplot(fig)
