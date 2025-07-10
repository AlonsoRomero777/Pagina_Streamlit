import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

def random_step(x, mu=0, sigma=0.1):
    return x + np.random.normal(mu, sigma, size=len(x))

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
    "Rastrigin": lambda x: rastrigin([x, x]),
    "Rosenbrock": lambda x: rosenbrock([x, x]),
    "Ackley": lambda x: ackley([x, x]),
    "Sphere": lambda x: sphere([x]),
    "Beale": lambda x: beale([x, x]),
    "Booth": lambda x: booth([x, x]),
    "Himmelblau": lambda x: himmelblau([x, x]),
    "McCormick": lambda x: mccormick([x, x]),
}


def busqueda_unidireccional(x0, delta, epsilon, funcion, max_iter=100):
    puntos = [(x0, funcion(x0))]
    for _ in range(max_iter):
        f0 = funcion(x0)
        x1 = x0 + delta
        x2 = x0 - delta
        f1 = funcion(x1)
        f2 = funcion(x2)

        if f1 < f0:
            x0 = x1
        elif f2 < f0:
            x0 = x2
        else:
            break

        puntos.append((x0, funcion(x0)))

        if abs(puntos[-1][1] - f0) < epsilon:
            break

    return x0, puntos


def plot_function_with_points(func, a, b, points, title):
    x = np.linspace(a, b, 1000)
    y = [func(xi) for xi in x]

    fig, ax = plt.subplots()
    ax.plot(x, y, label='Función', color='blue')
    if points:
        x_points, y_points = zip(*points)
        ax.scatter(x_points, y_points, color='red', label='Puntos visitados')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True)
    return fig



st.title("Búsqueda Unidireccional")

st.markdown("""
<style>
.justify-text {
    text-align: justify;
}
</style>

<div class="justify-text">
<p>Una <strong>búsqueda unidireccional</strong> es una técnica de optimización que consiste en buscar el valor mínimo 
(o máximo) de una función de una sola variable a lo largo de una dirección específica. Esta dirección es 
determinada previamente y se asume como fija durante el proceso de búsqueda.</p>

<p>Usualmente, una búsqueda unidireccional se efectúa desde un punto <strong>x(t)</strong> y en una dirección especificada
<strong>s(t)</strong>. Esto es, sólo se consideran en el proceso de búsqueda aquellos puntos que yacen sobre una línea 
(en un espacio N-dimensional, donde N es el número de variables de decisión del problema) que pasa a través del punto 
<strong>x(t)</strong> y está orientada a lo largo de la dirección de búsqueda <strong>s(t)</strong>.</p>

<p>El proceso implica transformar el problema multidimensional en un problema unidimensional, definiendo una nueva 
función escalar <strong>ϕ(α) = f(x + α·s)</strong>, donde <strong>α</strong> es el parámetro escalar que representa 
el avance sobre la dirección de búsqueda. A partir de esta función, se pueden aplicar diversos métodos unidimensionales, 
como la búsqueda dorada, bisección o métodos basados en gradiente.</p>

<h4>✔️ Propósito:</h4>
<ul>
<li>Encontrar el tamaño óptimo de paso en una dirección dada.</li>
<li>Servir como subrutina en métodos más complejos de optimización, como el descenso por gradiente.</li>
<li>Reducir un problema de múltiples variables a un problema más simple de una sola variable.</li>
</ul>

<h4>✔️ Ventajas:</h4>
<ul>
<li>Fácil de implementar.</li>
<li>Reduce la dimensionalidad del problema durante el proceso de búsqueda.</li>
<li>Útil como parte de algoritmos de optimización multivariables.</li>
</ul>

<h4>❌ Desventajas:</h4>
<ul>
<li>Requiere que la dirección de búsqueda sea apropiadamente seleccionada.</li>
<li>No garantiza encontrar el mínimo global si la dirección no apunta hacia él.</li>
<li>En problemas mal condicionados, puede requerir muchas iteraciones.</li>
</ul>

<h4>🔁 Relación con otros métodos:</h4>
<p>La búsqueda unidireccional es frecuentemente utilizada dentro de algoritmos iterativos como el 
<strong>descenso del gradiente</strong>, <strong>Newton-Raphson</strong> o el <strong>método de dirección conjugada</strong>, 
donde después de definir una dirección, se busca el valor óptimo del parámetro α en esa dirección.</p>

</div>
""", unsafe_allow_html=True)

opcion = st.selectbox("**Selecciona la función a optimizar**", list(funciones.keys()))
funcion = funciones[opcion]

x0 = st.number_input("**Valor inicial x₀**", value=1.0, step=0.1)
a = st.number_input("**Límite inferior (a)**", value=0.1, step=0.1)
b = st.number_input("**Límite superior (b)**", value=5.0, step=0.1)
delta = st.number_input("**Paso Δ**", value=0.1, step=0.01)
epsilon = st.number_input("**Precisión ε**", value=0.001, step=0.0001, format="%.4f")
max_iter = st.slider("**Máximo de iteraciones**", min_value=10, max_value=200, value=100)

if st.button("Ejecutar búsqueda"):
    if a >= b or x0 < a or x0 > b:
        st.error("El valor inicial debe estar dentro del intervalo [a, b] y a < b.")
    else:
        x_min, puntos = busqueda_unidireccional(x0, delta, epsilon, funcion, max_iter)
        st.success(f"Mínimo aproximado en x ≈ {x_min:.6f}, f(x) ≈ {funcion(x_min):.6f}")
        fig = plot_function_with_points(funcion, a, b, puntos, f"Búsqueda Unidireccional - {opcion}")
        st.pyplot(fig)
