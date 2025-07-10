import streamlit as st
import math
import matplotlib.pyplot as plt
import Fase_De_Acotamiento as BP
import time


def lata(r: float) -> float:
    return 2 * math.pi * r * r + (500 / r)

def caja(l: float) -> float:
    return -(4 * pow(l, 3) - 60 * l * l + 200 * l)

def funcion_0(x: float) -> float:
    if x == 0:
        return float('inf')
    return x**2 + (54/x)

def funcion_1(x: float) -> float:
    return x**3 + 2*x - 3

def funcion_2(x: float) -> float:
    return x**4 + x**2 - 33

def funcion_3(x: float) -> float:
    return 3*x**4 - 8*x**3 - 6*x**2 + 12*x


def Secante(a: float, b: float, epsilon: float, funcion: callable):
    fa = funcion(a)
    fb = funcion(b)
    z = (a + b) / 2
    px = [z]
    py = [funcion(z)]

    grafico_placeholder = st.empty()

    while abs(funcion(z)) > epsilon and (z > a) and (z < b):
        if funcion(z) < 0:
            a = z
            fa = funcion(a)
        else:
            b = z
            fb = funcion(b)

        if fb - fa == 0:  
            break

        z = b - (fb * (b - a)) / (fb - fa)
        px.append(z)
        py.append(funcion(z))

        fig, ax = plt.subplots()
        ax.plot(px, py, label='Evaluaciones')
        ax.scatter(px, py, c="red", label='Puntos visitados')
        ax.set_title("Método de la Secante")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid(True)
        grafico_placeholder.pyplot(fig)

        time.sleep(0.3)  

    return z, px, py


st.title("Método de la Secante")

texto_justificado = """
<div style="text-align: justify;">
<p>El <strong>método de la secante</strong> es una técnica numérica utilizada para encontrar raíces de funciones no lineales. 
A diferencia del método de Newton-Raphson, no requiere calcular derivadas analíticas, lo que lo convierte en una alternativa útil 
cuando la derivada de la función es difícil o costosa de obtener.</p>

<p>Este método se basa en aproximar la derivada mediante la pendiente de una línea secante trazada entre dos puntos previos 
de la función. A partir de estos dos valores, se calcula una intersección con el eje x, generando una nueva estimación de la raíz. 
La fórmula utilizada es:</p>

<p style="text-align: center;"><code>xₙ₊₁ = xₙ - f(xₙ) · (xₙ - xₙ₋₁) / (f(xₙ) - f(xₙ₋₁))</code></p>

<p>Este proceso se repite de manera iterativa, utilizando siempre los dos valores más recientes para construir una nueva secante 
y generar una mejor aproximación. El algoritmo termina cuando la diferencia entre las estimaciones sucesivas es menor que 
una tolerancia establecida o cuando el valor de la función es suficientemente cercano a cero.</p>

<p><strong>Ventajas:👍</strong></p>
<ul>
<li>No requiere derivadas analíticas.</li>
<li>Convergencia más rápida que el método de bisección (superlineal).</li>
<li>Fácil de implementar con solo dos evaluaciones por iteración.</li>
</ul>

<p><strong>Desventajas:👎</strong></p>
<ul>
<li>No siempre garantiza convergencia si los puntos iniciales no están bien elegidos.</li>
<li>Puede divergir si la función es muy irregular o presenta oscilaciones.</li>
<li>No tiene la misma velocidad de convergencia cuadrática del método de Newton-Raphson en condiciones ideales.</li>
</ul>

<p>El método de la secante es una opción eficiente y práctica cuando no se dispone de la derivada de la función, 
combinando simplicidad computacional con una velocidad aceptable de convergencia en la mayoría de los casos.</p>
</div>
"""

st.markdown(texto_justificado, unsafe_allow_html=True)

funciones = {
    "Lata": (lata, 0.2, 5),
    "Caja": (caja, 0.2, 5),
    "Función 0: x² + 54/x": (funcion_0, 0.5, 5),
    "Función 1: x³ + 2x - 3": (funcion_1, 0.5, 5),
    "Función 2: x⁴ + x² - 33": (funcion_2, -2.5, 2.5),
    "Función 3: 3x⁴ - 8x³ - 6x² + 12x": (funcion_3, -1.5, 3),
}

opcion = st.selectbox("Selecciona una función:", list(funciones.keys()))
funcion, x0_default, d_default = funciones[opcion]

x0 = st.number_input("Valor inicial x₀", value=x0_default, step=0.1)
delta = st.number_input("Valor de incremento Δ", value=float(d_default), step=0.1)
epsilon = st.number_input("Tolerancia ε", min_value=0.0001, value=0.001, step=0.0001, format="%.4f")

if st.button("Ejecutar"):
    try:
        intervalo = BP.fase_acotamiento(x0, delta, 0.1, funcion)
        st.success(f"Intervalo encontrado: {intervalo}")

        minimo, px, py = Secante(*intervalo, epsilon, funcion)
        st.write(f"Resultado mínimo: x = {minimo:.5f}, f(x) = {funcion(minimo):.5f}")

        fig, ax = plt.subplots()
        ax.plot(px, py, label='Evaluaciones')
        ax.scatter(px, py, c="red", label='Puntos visitados')
        ax.set_title("Método de la Secante")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid(True)


    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
