import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


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

funciones_dict = {
    "Booth": booth,
    "Beale": beale,
    "Himmelblau": himmelblau,
    "McCormick": mccormick,
    "Sphere": sphere,
    "Rosenbrock": rosenbrock,
    "Ackley": ackley,
    "Rastrigin": rastrigin,
}

limites_dict = {
    "Booth": (-10, 10),
    "Beale": (-4.5, 4.5),
    "Himmelblau": (-6, 6),
    "McCormick": (-2, 4),
    "Sphere": (-5, 5),
    "Rosenbrock": (-2, 2),
    "Ackley": (-5, 5),
    "Rastrigin": (-5.12, 5.12),
}


def meshdata(x_min, x_max, y_min, y_max, function, n_puntos=200):
    x_vals = np.linspace(x_min, x_max, n_puntos)
    y_vals = np.linspace(y_min, y_max, n_puntos)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.array([function(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)
    return X, Y, Z

def movimiento_exploratorio(x, delta, funcion, N):
    xc = np.array(x)
    for i in range(N):
        x_plus = np.array(x)
        x_minus = np.array(x)
        x_plus[i] = x[i] + delta[i]
        x_minus[i] = x[i] - delta[i]
        xs = [x_minus, x, x_plus]
        fx = [funcion(x) for x in xs]
        x = xs[np.argmin(fx)]
    if np.allclose(x, xc):
        return xc, False
    return x, True

def hooke_jeeves(funcion, x0, delta, epsilon=1e-5, alpha=2.0, N=2, max_iter=100):
    x = np.array(x0)
    historial = [x.copy()]
    for _ in range(max_iter):
        xn, mov = movimiento_exploratorio(x, delta, funcion, N)
        if not mov:
            delta = delta / 2.0
            if np.all(delta < epsilon):
                break
        else:
            xp = xn + alpha * (xn - x)
            fxp = funcion(xp)
            fx = funcion(x)
            if fxp < fx:
                x = xp
            else:
                x = xn
        historial.append(x.copy())
    return x, historial

def plotContourWithPath(X, Y, Z, path):
    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, levels=30, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], marker='o', color='red')
    ax.set_title('GRAFICA DE CONTORNO')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return fig

st.title("Método de Optimización de Hooke-Jeeves")

st.markdown("""
El **método de Hooke-Jeeves** es un algoritmo de optimización sin derivadas que combina dos tipos de movimientos para buscar el mínimo de una función multivariable:

- 🔍 **Movimiento exploratorio**: se exploran pequeñas variaciones en cada variable individualmente para detectar mejoras locales.
- 🧭 **Movimiento de patrón**: si se encuentra una mejora en la fase exploratoria, se da un salto en esa dirección con la esperanza de acercarse rápidamente al óptimo.

Este método es útil cuando la función no es diferenciable, es costosa de evaluar o no se dispone del gradiente. Aquí puedes probarlo en varias funciones clásicas.
""")

st.subheader("Selecciona la función objetivo")
nombre_funcion = st.selectbox("Funciones disponibles:", list(funciones_dict.keys()))
funcion_objetivo = funciones_dict[nombre_funcion]
x_min, x_max = limites_dict[nombre_funcion]
y_min, y_max = limites_dict[nombre_funcion]

st.subheader("Parámetros de entrada")
col1, col2 = st.columns(2)
with col1:
    x0_0 = st.number_input("x0[0] (punto inicial)", value=0.0)
    delta_0 = st.number_input("delta[0] (paso inicial)", value=1.0)
    epsilon = st.number_input("Epsilon (tolerancia)", value=0.001)
with col2:
    x0_1 = st.number_input("x0[1] (punto inicial)", value=0.0)
    delta_1 = st.number_input("delta[1] (paso inicial)", value=1.0)
    alpha = st.number_input("Alpha (factor de patrón)", value=2.0)

max_iter = st.slider("Máximo de iteraciones", 10, 500, 100)

if st.button("Ejecutar Hooke-Jeeves"):
    x0 = [x0_0, x0_1]
    delta = np.array([delta_0, delta_1])
    X, Y, Z = meshdata(x_min, x_max, y_min, y_max, funcion_objetivo)

    sol, path = hooke_jeeves(funcion_objetivo, x0, delta, epsilon=epsilon, alpha=alpha, N=2, max_iter=max_iter)
    fig = plotContourWithPath(X, Y, Z, path)

    st.pyplot(fig)

    st.success(f"Punto mínimo encontrado: {np.round(sol, 6)}")
    st.info(f"Valor de la función en el mínimo: {round(funcion_objetivo(sol), 6)}")
