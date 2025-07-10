import streamlit as st
import numpy as np
import matplotlib.pyplot as plt




st.title("Algoritmo de Optimización: Random Walk")

st.markdown("""
    <style>
    .justify {
        text-align: justify;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="justify">
<p><strong>Random Walk</strong> (Caminata Aleatoria) es un algoritmo de optimización estocástico que consiste en explorar el espacio de búsqueda moviéndose de manera aleatoria desde una solución actual hacia nuevas soluciones vecinas. Si la nueva solución mejora el valor de la función objetivo, se acepta como la nueva solución actual. Este proceso se repite durante un número determinado de iteraciones.</p>

<p>El principio detrás de este método es simple: permitir que el algoritmo se desplace sin una dirección fija, lo que le da la posibilidad de explorar diferentes regiones del espacio de búsqueda sin sesgo. Esto lo convierte en una estrategia útil para problemas donde no se dispone de gradientes o cuando el paisaje de la función objetivo es muy irregular.</p>

<p>Aunque por sí solo puede ser ineficiente para encontrar el óptimo global, <strong>Random Walk</strong> es frecuentemente utilizado como componente dentro de algoritmos más complejos como el Recocido Simulado, Algoritmos Evolutivos o Algoritmos Genéticos, para introducir diversidad y evitar estancamientos en mínimos locales.</p>

<p>Algunas características importantes del algoritmo:</p>
<ul>
<li><strong>Simplicidad:</strong> Es fácil de implementar y no requiere derivadas ni estructura del problema.</li>
<li><strong>Exploración aleatoria:</strong> Las soluciones se generan mediante desplazamientos aleatorios controlados por una magnitud o paso.</li>
<li><strong>No direccionalidad:</strong> A diferencia del gradiente, no se busca una dirección óptima, lo cual puede ser útil en funciones no diferenciables o ruidosas.</li>
<li><strong>Limitaciones:</strong> Tiene baja eficiencia en espacios grandes y sin guía puede requerir muchas iteraciones para obtener buenos resultados.</li>
</ul>

<p>Random Walk es una técnica básica pero poderosa para la exploración inicial de funciones objetivo complejas, y puede ser mejorada significativamente cuando se combina con estrategias de aceptación inteligentes o mecanismos de memoria.</p>
</div>
""", unsafe_allow_html=True)



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
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock,
    "Ackley": ackley,
    "Sphere": sphere,
    "Beale": beale,
    "Booth": booth,
    "Himmelblau": himmelblau,
    "McCormick": mccormick,
}

opcion = st.selectbox("Selecciona una función objetivo", list(funciones.keys()))

max_iter = st.slider("Número de iteraciones", 100, 5000, 1000, step=100)
sigma = st.slider("Tamaño del paso (sigma)", 0.01, 1.0, 0.3)

def random_walk(f, x0, max_iter=1000, mu=0, sigma=0.1):
    x_best = np.copy(x0)
    f_best = f(x_best)
    history = [f_best]
    path = [x_best.copy()]

    for _ in range(max_iter):
        x_new = random_step(x_best, mu, sigma)
        f_new = f(x_new)

        if f_new < f_best:
            x_best = x_new
            f_best = f_new

        history.append(f_best)
        path.append(x_best.copy())

    return x_best, f_best, history, path

np.random.seed(42)  
x0 = np.random.uniform(-5, 5, size=2)

func = funciones[opcion]
sol, val, hist, path = random_walk(func, x0, max_iter=max_iter, sigma=sigma)


st.markdown(f"**Mejor solución encontrada:** {sol}")
st.markdown(f"**Valor mínimo de la función:** {val:.6f}")

path = np.array(path)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(path[:, 0], path[:, 1], marker='o', markersize=2, linestyle='-', label='Camino')
ax.scatter(path[0, 0], path[0, 1], color='green', label='Inicio', s=50)
ax.scatter(path[-1, 0], path[-1, 1], color='red', label='Fin', s=50)
ax.set_title(f'Recorrido en la función {opcion}')
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.grid()
ax.legend()
st.pyplot(fig)
