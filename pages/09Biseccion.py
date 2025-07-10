import streamlit as st
import math
import matplotlib.pyplot as plt
import Fase_De_Acotamiento as BP
import time


def lata(r: float) -> float:
    if r == 0:
        return float('inf')
    return 2 * math.pi * r * r + (500 / r)

def caja(l: float) -> float:
    return -(4 * l**3 - 60 * l**2 + 200 * l)

def funcion_0(x: float) -> float:
    if x == 0:
        return float('inf')
    return x**2 + (54 / x)

def funcion_1(x: float) -> float:
    return x**3 + 2 * x - 3

def funcion_2(x: float) -> float:
    return x**4 + x**2 - 33

def funcion_3(x: float) -> float:
    return 3 * x**4 - 8 * x**3 - 6 * x**2 + 12 * x



def primeraDerivada(x: float, delta: float, funcion: callable):
    return ((funcion(x + delta) - funcion(x - delta)) / (2 * delta))


def Biseccion(a: float, b: float, epsilon: float, funcion: callable):
    z = (a + b) / 2
    funcionz = primeraDerivada(z, epsilon, funcion)

    px = [z]
    py = [funcion(z)]

    grafico_placeholder = st.empty()

    while abs(funcionz) > epsilon and (z > a) and (z < b):
        if funcionz < 0:
            a = z
        else:
            b = z

        z = (a + b) / 2
        funcionz = primeraDerivada(z, epsilon, funcion)
        px.append(z)
        py.append(funcion(z))

        fig, ax = plt.subplots()
        ax.plot(px, py, label='Evaluaciones')
        ax.scatter(px, py, color='red', label='Puntos visitados')
        ax.set_title("Método de Bisección")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True)
        ax.legend()
        grafico_placeholder.pyplot(fig)

        time.sleep(0.3)  

    return z, px, py



st.title("Método de Bisección")

texto_justificado = """
<div style="text-align: justify;">
<p>El <strong>método de bisección</strong> es una técnica numérica simple y robusta para encontrar raíces de funciones continuas. 
Se basa en el <strong>Teorema del Valor Intermedio</strong>, el cual establece que si una función continua cambia de signo 
en un intervalo [a, b] (es decir, si <i>f(a) · f(b) &lt; 0</i>), entonces existe al menos una raíz en ese intervalo.</p>

<p>El algoritmo consiste en dividir el intervalo a la mitad en cada iteración y evaluar la función en el punto medio. 
Según el signo de la función en ese punto, se elige el subintervalo en el que persiste el cambio de signo. 
Este proceso se repite sucesivamente hasta que el tamaño del intervalo sea suficientemente pequeño, garantizando que la raíz 
esté contenida dentro de dicho intervalo con una precisión deseada.</p>

<p><strong>Ventajas:👍</strong></p>
<ul>
<li>Muy fácil de implementar y comprender.</li>
<li>Garantiza convergencia si se cumple la condición de cambio de signo.</li>
<li>No requiere derivadas ni información adicional de la función.</li>
<li>Es útil con funciones no suaves, no derivables o ruidosas.</li>
</ul>

<p><strong>Desventajas:👎</strong></p>
<ul>
<li>Convergencia lenta (lineal) en comparación con métodos como Newton-Raphson o la secante.</li>
<li>No se puede aplicar si la función no cambia de signo en el intervalo inicial.</li>
<li>No identifica múltiples raíces dentro del mismo intervalo.</li>
</ul>

<p>El método de bisección es ideal cuando se busca seguridad y robustez en la localización de raíces, especialmente en 
problemas donde otras técnicas pueden fallar por falta de derivadas, mal comportamiento de la función, o sensibilidad a 
la elección del punto inicial.</p>
</div>
"""

st.markdown(texto_justificado, unsafe_allow_html=True)


funciones = {
    "Lata": (lata, 0.2, 5.0),
    "Caja": (caja, 0.2, 5.0),
    "Función 0: x² + 54/x": (funcion_0, 0.5, 5.0),
    "Función 1: x³ + 2x - 3": (funcion_1, 0.5, 5.0),
    "Función 2: x⁴ + x² - 33": (funcion_2, -2.5, 2.5),
    "Función 3: 3x⁴ - 8x³ - 6x² + 12x": (funcion_3, -1.5, 3.0),
}


opcion = st.selectbox("Selecciona una función:", list(funciones.keys()))
funcion, x0_default, d_default = funciones[opcion]

x0 = st.number_input("Valor inicial x₀", value=x0_default, step=0.1)
delta = st.number_input("Valor de incremento Δ", value=d_default, step=0.1)
epsilon = st.number_input("Tolerancia ε", min_value=0.0001, value=0.001, step=0.0001, format="%.4f")

if st.button("Ejecutar"):
    try:
        intervalo = BP.fase_acotamiento(x0, delta, 0.1, funcion)
        st.success(f"Intervalo encontrado: {intervalo}")

        minimo, px, py = Biseccion(*intervalo, epsilon, funcion)
        st.write(f"Resultado mínimo: x = {minimo:.5f}, f(x) = {funcion(minimo):.5f}")

        fig, ax = plt.subplots()
        ax.plot(px, py, label='Evaluaciones')
        ax.scatter(px, py, color='red', label='Puntos visitados')
        ax.set_title("Método de Bisección")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True)
        ax.legend()
        
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
