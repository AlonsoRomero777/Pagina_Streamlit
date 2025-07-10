import numpy as np
import matplotlib.pyplot as plt
import math
import streamlit as st
import time

def lata(r: float) -> float:
    """Calcula el área de una lata en función de su radio."""
    if r == 0:
        return float('inf')  
    return 2 * math.pi * r * r + (500 / r)

def caja(l: float) -> float:
    return -(4 * pow(l, 3) - 60 * l * l + 200 * l)

def funcion_0(x: float) -> float:
    return float('inf') if x == 0 else x**2 + (54/x)

def funcion_1(x: float) -> float:
    return x**3 + 2*x - 3

def funcion_2(x: float) -> float:
    return x**4 + x**2 - 33

def funcion_3(x: float) -> float:
    return 3*x**4 - 8*x**3 - 6*x**2 + 12*x

funciones = {
    "Lata (área)": lata,
    "Caja (función negativa)": caja,
    "x² + 54/x": funcion_0,
    "x³ + 2x - 3": funcion_1,
    "x⁴ + x² - 33": funcion_2,
    "3x⁴ - 8x³ - 6x² + 12x": funcion_3
}


def newton_raphson(f, x0, epsilon, delta_x, max_iter=100, a=None, b=None):
    points = []
    x_old = x0
    points.append((x_old, f(x_old)))

    grafico_placeholder = st.empty()
    rango_grafica = (a, b)

    for i in range(max_iter):
        f_plus = f(x_old + delta_x)
        f_minus = f(x_old - delta_x)
        derivative = (f_plus - f_minus) / (2 * delta_x)

        if abs(derivative) < 1e-10:
            break

        x_new = x_old - f(x_old) / derivative

        if a is not None and x_new < a:
            x_new = a
        if b is not None and x_new > b:
            x_new = b

        points.append((x_new, f(x_new)))

        fig = plot_function_with_points(f, rango_grafica[0], rango_grafica[1], points, f"Newton-Raphson - {opcion}")
        grafico_placeholder.pyplot(fig)
        time.sleep(0.5) 

        if abs(x_new - x_old) < epsilon or abs(f(x_new)) < epsilon:
            break

        x_old = x_new

    return x_old, points


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
    ax.grid(True)
    ax.legend()
    return fig

st.title("Método de Newton-Raphson")

texto_justificado = """
<div style="text-align: justify;">
<p>El <strong>método de Newton-Raphson</strong> es una técnica iterativa utilizada para encontrar los mínimos o máximos locales 
de una función diferenciable. A diferencia de su uso en la resolución de ecuaciones no lineales (donde se busca 
resolver <i>f(x) = 0</i>), en el contexto de la optimización se emplea para encontrar puntos críticos, es decir, aquellos 
donde la primera derivada de la función es cero: <i>f'(x) = 0</i>.</p>

<p>Este método aprovecha tanto la primera como la segunda derivada de la función objetivo. A partir de un punto inicial <i>x₀</i>, 
el método genera una secuencia de puntos mediante la fórmula:</p>

<p style="text-align: center;"><code>xₙ₊₁ = xₙ - f'(xₙ)/f''(xₙ)</code></p>

<p>Esta expresión ajusta el valor de <i>x</i> en cada paso, usando la pendiente y la curvatura de la función para avanzar hacia un punto 
donde la pendiente sea cero (óptimo local).</p>

<p><strong>Ventajas:👍</strong></p>
<ul>
<li>Alta velocidad de convergencia cuando el punto inicial está cerca del mínimo o máximo.</li>
<li>Requiere pocas iteraciones en funciones suaves y bien comportadas.</li>
<li>Ideal para funciones diferenciables con buena curvatura.</li>
</ul>

<p><strong>Desventajas:👎</strong></p>
<ul>
<li>Requiere calcular la segunda derivada de la función, lo cual puede ser costoso o complejo.</li>
<li>Si <i>f''(x)</i> es cero o muy cercano a cero, puede provocar divisiones por cero o inestabilidad numérica.</li>
<li>No garantiza convergencia si el punto inicial está lejos del óptimo o si la función no es convexa.</li>
<li>Puede converger a un punto que no sea un mínimo, sino un máximo o un punto de silla.</li>
</ul>

<p>Newton-Raphson es una herramienta poderosa para la optimización local cuando se cuenta con información 
analítica de la función y se parte desde un punto cercano al óptimo. Es comúnmente usado como parte de métodos más avanzados, 
como el descenso de Newton modificado o algoritmos cuasi-Newton.</p>
</div>
"""

st.markdown(texto_justificado, unsafe_allow_html=True)


opcion = st.selectbox("**Selecciona la función a optimizar**", list(funciones.keys()))
funcion = funciones[opcion]

x0 = st.number_input("**Valor inicial x₀**", value=1.0, key="x0_nr")
a = st.number_input("**Límite inferior (a)**", value=0.0, key="a_nr")
b = st.number_input("**Límite superior (b)**", value=5.0, key="b_nr")
epsilon = st.number_input("**Precisión (ε)**", value=0.001, format="%.6f")
delta_x = st.number_input("**Delta x (para derivada)**", value=0.01, format="%.6f")
max_iter = st.slider("**Máximo de iteraciones**", min_value=10, max_value=200, value=100)

if st.button("Ejecutar Newton-Raphson"):
    if a >= b:
        st.error("El límite inferior debe ser menor que el superior.")
    else:
        minimo, puntos = newton_raphson(funcion, x0, epsilon, delta_x, max_iter, a, b)
        st.success(f"Mínimo aproximado: x ≈ {minimo:.6f}, f(x) ≈ {funcion(minimo):.6f}")

