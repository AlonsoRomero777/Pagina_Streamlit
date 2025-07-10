import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def rosenbrock(x):
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

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

def createSimplex(x0: np.ndarray, alpha: float, N: int):
    delta1 = ((np.sqrt(N+1) + N - 1) / (N * np.sqrt(2))) * alpha
    delta2 = ((np.sqrt(N+1) - 1) / (N * np.sqrt(2))) * alpha
    xn = [np.array([x0[i] + delta1 if i == j else x0[i] + delta2 for i in range(N)]) for j in range(N)]
    xn.insert(0, x0)
    return np.array(xn)

def terminar(fx, fc, N, epsilon=0.001):
    return np.sqrt(np.sum(((fx - fc)**2)/(N+1))) < epsilon

def plotContourWithSimplex(X, Y, Z, simplex, nuevo_punto=None):
    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, levels=30, cmap='viridis')
    fig.colorbar(contour, ax=ax)

    for punto in simplex:
        ax.plot(punto[0], punto[1], 'r*', markersize=10)

    if nuevo_punto is not None:
        ax.plot(nuevo_punto[0], nuevo_punto[1], 'bs', markersize=8)

    ax.set_title('GRAFICA DE CONTORNO')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return fig


st.title("Método de Nelder-Mead (Simplex)")

st.markdown("""
El **método de Nelder-Mead** es un algoritmo de optimización numérica utilizado para encontrar mínimos de funciones multivariables sin derivadas. Usa transformaciones del simplex para mejorar en cada iteración.

Este método es útil para funciones no diferenciables o ruidosas. Aquí puedes probarlo con distintas funciones clásicas.
""")

st.subheader("Selecciona la función objetivo")
nombre_funcion = st.selectbox("Funciones disponibles:", list(funciones_dict.keys()))
funcion_objetivo = funciones_dict[nombre_funcion]
x_min, x_max = limites_dict[nombre_funcion]
y_min, y_max = limites_dict[nombre_funcion]


col1, col2 = st.columns(2)
with col1:
    x0_0 = st.number_input("x0[0]", value=0.0)
    alpha = st.number_input("Alpha (tamaño simplex)", value=1.0)
    epsilon = st.number_input("Epsilon (criterio de parada)", value=0.001)
with col2:
    x0_1 = st.number_input("x0[1]", value=0.0)
    gamma = st.number_input("Gamma (expansión)", value=1.2)
    beta = st.number_input("Beta (contracción)", value=0.5)

max_iter = st.slider("Máximo de iteraciones", min_value=1, max_value=200, value=100)

if st.button("Ejecutar Nelder-Mead"):
    N = 2
    x0 = np.array([x0_0, x0_1])
    simplex = createSimplex(x0, alpha, N)
    trayectoria = [simplex.copy()]
    final = False
    iteracion = 0

    while not final and iteracion < max_iter:
        fx = np.array([funcion_objetivo(x) for x in simplex])
        indices = np.argsort(fx)
        i_xl, i_xg, i_xh = indices[0], indices[-2], indices[-1]

        xc = (np.sum(simplex, axis=0) - simplex[i_xh]) / N
        xr = 2 * xc - simplex[i_xh]
        fxr = funcion_objetivo(xr)
        fxc = funcion_objetivo(xc)
        xnew = xr

        if fxr < fx[i_xl]:
            xnew = (1 + gamma)*xc - gamma*simplex[i_xh]  
        elif fxr >= fx[i_xh]:
            xnew = (1 - beta)*xc + beta*simplex[i_xh]  
        elif fx[i_xg] < fxr < fx[i_xh]:
            xnew = (1 + beta)*xc - beta*simplex[i_xh] 

        simplex[i_xh] = xnew
        trayectoria.append(simplex.copy())
        final = terminar(fx, fxc, N, epsilon)
        iteracion += 1

    X, Y, Z = meshdata(x_min, x_max, y_min, y_max, funcion_objetivo)
    fig = plotContourWithSimplex(X, Y, Z, simplex)
    st.pyplot(fig)

    mejor = simplex[np.argmin([funcion_objetivo(x) for x in simplex])]
    st.success(f"Punto mínimo encontrado: {np.round(mejor, 6)}")
    st.info(f"Valor de la función: {round(funcion_objetivo(mejor), 6)}")
    st.write(f"Iteraciones realizadas: {iteracion}")
