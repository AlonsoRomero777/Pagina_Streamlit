import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.markdown("""
<style>
.justify { text-align: justify; }
</style>
""", unsafe_allow_html=True)

st.title("Método de Newton")

st.markdown("""
<div class="justify">
<p><strong>El Método de Newton</strong> es un algoritmo de optimización de segundo orden ampliamente utilizado para encontrar mínimos (o máximos) locales de funciones diferenciables. Se basa en una aproximación cuadrática de la función objetivo mediante una expansión de Taylor de segundo orden.</p>

<p>La fórmula iterativa del método es:</p>
<center>
x<sub>k+1</sub> = x<sub>k</sub> - H<sup>-1</sup>(x<sub>k</sub>) ⋅ ∇f(x<sub>k</sub>)
</center>

<p>donde ∇f(x) es el gradiente de la función y H(x) es la matriz Hessiana evaluada en el punto x. Esta fórmula asume que la matriz Hessiana es invertible y que la función es al menos dos veces continuamente diferenciable.</p>

<p><strong>Ventajas: 👍</strong></p>
<ul>
<li>Convergencia cuadrática cerca del mínimo: el método converge muy rápidamente cuando se encuentra cerca del óptimo.</li>
<li>Puede encontrar mínimos sin necesidad de un paso de búsqueda explícito, si la Hessiana es bien condicionada.</li>
</ul>

<p><strong>Desventajas: 👎</strong></p>
<ul>
<li>El cálculo de la matriz Hessiana puede ser costoso para funciones de muchas variables.</li>
<li>Si la Hessiana no es definida positiva (es decir, si tiene valores negativos en sus autovalores), el método puede no converger al mínimo.</li>
<li>Requiere un buen punto inicial, ya que podría converger a un punto de silla o máximo si se usa incorrectamente.</li>
</ul>

<p>En aplicaciones reales, se suele usar una versión modificada llamada <strong>método de Newton modificado</strong> o <strong>cuasi-Newton</strong>, como los métodos BFGS y L-BFGS, que aproximan la Hessiana para mejorar la eficiencia.</p>

<p>Este método se recomienda cuando se desea una solución precisa y se tiene acceso a derivadas exactas o se pueden calcular de manera eficiente.</p>
</div>
""", unsafe_allow_html=True)




def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def rosenbrock(x):
    return sum([100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1)])

def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def sphere(x):
    return sum([xi**2 for xi in x])

def booth(x):
    x1, x2 = x[0], x[1]
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

def beale(x):
    x1, x2 = x[0], x[1]
    return ((1.5 - x1 + x1 * x2)**2 + 
            (2.25 - x1 + x1 * x2**2)**2 + 
            (2.625 - x1 + x1 * x2**3)**2)

def ackley(x):
    a, b, c = 20, 0.2, 2*np.pi
    d = len(x)
    sum1 = sum([xi**2 for xi in x])
    sum2 = sum([np.cos(c * xi) for xi in x])
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

def mccormick(x):
    x1, x2 = x[0], x[1]
    return np.sin(x1 + x2) + (x1 - x2)**2 - 1.5 * x1 + 2.5 * x2 + 1

funciones = {
    "Himmelblau": himmelblau,
    "Rosenbrock": rosenbrock,
    "Rastrigin": rastrigin,
    "Sphere": sphere,
    "Booth": booth,
    "Beale": beale,
    "Ackley": ackley,
    "McCormick": mccormick,
}

nombre_funcion = st.selectbox("Selecciona una función objetivo", list(funciones.keys()))
funcion = funciones[nombre_funcion]

epsilon1 = st.slider("Tolerancia para gradiente (ε₁)", 1e-6, 0.01, 0.001)
epsilon2 = st.slider("Tolerancia para cambio relativo (ε₂)", 1e-6, 0.01, 0.001)
M = st.slider("Máximo de iteraciones", 10, 1000, 200)



def gradiente(f, x, h=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

def hessian_matrix(f, x, deltaX=1e-5):
    fx = f(x)
    N = len(x)
    H = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                xp = x.copy()
                xnn = x.copy()
                xp[i] += deltaX
                xnn[i] -= deltaX
                H[i, j] = (f(xp) - 2*fx + f(xnn)) / (deltaX**2)
            else:
                xpp = x.copy()
                xpn = x.copy()
                xnp = x.copy()
                xnn = x.copy()
                xpp[i] += deltaX
                xpp[j] += deltaX
                xpn[i] += deltaX
                xpn[j] -= deltaX
                xnp[i] -= deltaX
                xnp[j] += deltaX
                xnn[i] -= deltaX
                xnn[j] -= deltaX
                H[i, j] = (f(xpp) - f(xpn) - f(xnp) + f(xnn)) / (4 * deltaX**2)
    return H



def newton(f, x0, epsilon1, epsilon2, M):
    k = 0
    xk = x0
    path = [xk.copy()]
    while True:
        grad = gradiente(f, xk)
        hess = hessian_matrix(f, xk)
        if np.linalg.norm(grad) < epsilon1 or k > M:
            break
        try:
            hess_inv = np.linalg.inv(hess)
            xk1 = xk - hess_inv @ grad
        except np.linalg.LinAlgError:
            xk1 = xk - 0.01 * grad

        if np.linalg.norm(xk1 - xk) / (np.linalg.norm(xk) + 1e-10) < epsilon2:
            break
        xk = xk1
        path.append(xk.copy())
        k += 1
    return xk, path

np.random.seed(42)
x0 = np.random.uniform(-4, 4, size=2)
resultado, tray = newton(funcion, x0, epsilon1, epsilon2, M)

st.markdown(f"**Mejor solución encontrada:** `{resultado}`")
st.markdown(f"**Valor mínimo estimado:** `{funcion(resultado):.6f}`")


tray = np.array(tray)
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)
ax.plot(tray[:, 0], tray[:, 1], marker='o', markersize=2, linestyle='-', alpha=0.7)
ax.scatter(tray[0, 0], tray[0, 1], color='green', label='Inicio', s=80)
ax.scatter(tray[-1, 0], tray[-1, 1], color='red', label='Fin', s=80)

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

st.markdown("---")
