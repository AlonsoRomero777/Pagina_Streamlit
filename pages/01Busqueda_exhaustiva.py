import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import time


def funcion_chida(x: float) -> float:
    x_alpha1 = 2 + x * 2
    x_alpha2 = 1 + x * 5
    return (x_alpha1 - 10)**2 + (x_alpha2 - 10)**2

def funcion_00(x: float) -> float:
    return x**2 + 3

def lata(r: float) -> float:
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



import time

def busqueda_exhaustiva_animada(a: float, b: float, n: int, funcion: callable):
    delta_x = (b - a) / n
    x1 = a
    x2 = x1 + delta_x
    x3 = x2 + delta_x

    valores_x = []
    valores_y = []

    grafico_placeholder = st.empty()

    while x3 <= b and not (funcion(x1) >= funcion(x2) <= funcion(x3)):
        valores_x.append(x2)
        valores_y.append(funcion(x2))

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(valores_x, valores_y, 'bo-', label='Puntos evaluados')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Búsqueda en [{a}, {b}]')
        ax.legend()
        grafico_placeholder.pyplot(fig)
        time.sleep(0.3)  

        x1 = x2
        x2 = x3
        x3 = x2 + delta_x

    return x1, x3


st.title("Búsqueda Exhaustiva 🔝")

st.markdown("""
<style>
.justify-text {
    text-align: justify;
}
</style>

<div class="justify-text">
<p>La <strong>búsqueda exhaustiva</strong> es uno de los métodos más simples pero efectivos en optimización unidimensional.
Consiste en evaluar una función en varios puntos distribuidos uniformemente dentro de un intervalo definido por el usuario, 
con el objetivo de identificar el punto que proporciona el valor mínimo (o máximo) de la función.</p>

<p>Este método parte de un <strong>límite inferior</strong> y un <strong>límite superior</strong> para la variable de decisión,
dividiendo el intervalo en segmentos de tamaño igual determinado por una precisión o incremento <strong>Δx</strong>. 
Luego, se evalúan los valores de la función en esos puntos y se comparan <strong>tres consecutivos</strong> a la vez 
para encontrar el óptimo, bajo el supuesto de que la función es <strong>unimodal</strong> (tiene un solo mínimo o máximo en el intervalo).</p>

<h4>👍 Ventajas:</h4>
<ul>
<li>Fácil de implementar y entender.</li>
<li>No requiere derivadas ni información adicional de la función.</li>
<li>Puede aplicarse a funciones discontinuas o con ruido.</li>
</ul>

<h4>👎 Desventajas:</h4>
<ul>
<li>Poco eficiente para funciones complejas o de alta dimensión.</li>
<li>Puede requerir muchas evaluaciones para alcanzar una buena precisión.</li>
<li>No garantiza encontrar el mínimo global si hay múltiples óptimos locales.</li>
</ul>

<h4> ¿Cuándo usarla?</h4>
<p>La búsqueda exhaustiva es ideal para problemas donde se desconoce la forma de la función, o cuando ésta 
es difícil de derivar o no es continua. También se usa para validar el comportamiento de otros métodos 
más sofisticados, al servir como <strong>punto de referencia</strong> por su simplicidad.</p>

<h4> ¿Qué puedes hacer en esta aplicación?</h4>
<ul>
<li>Seleccionar la función a optimizar.</li>
<li>Definir el intervalo de búsqueda (límites inferior y superior).</li>
<li>Ajustar la precisión mediante el parámetro <strong>Δx</strong>.</li>
<li>Visualizar el recorrido de puntos evaluados y el óptimo encontrado.</li>
</ul>

</div>
""", unsafe_allow_html=True)

funciones = {
    "Ejercicio 1: f(x) = x² + 3": (funcion_00, -2, 2),
    "Función Lata": (lata, 0.1, 10),
    "Función Caja": (caja, 2, 3),
    "Función 0: f(x) = x² + 54/x": (funcion_0, 0.1, 10),
    "Función 1: f(x) = x³ + 2x - 3": (funcion_1, 0, 5),
    "Función 2: f(x) = x⁴ + x² - 33": (funcion_2, -2.5, 2.5),
    "Función 3: f(x) = 3x⁴ - 8x³ - 6x² + 12x": (funcion_3, -1.5, 3),
    "Función Chida": (funcion_chida, 0, 5),
    "Función Personalizada": (None, -10, 10)
}

seleccion = st.selectbox("**Selecciona la función a optimizar:**", list(funciones.keys()))
funcion, a_def, b_def = funciones[seleccion]

if seleccion == "Función Personalizada":
    expresion = st.text_input("Escribe tu función en términos de x (ej. x**2 + 3*x - 1):", value="x**2")
    try:
        funcion = lambda x: eval(expresion, {"x": x, "math": math, "np": np})
        st.success("Función cargada correctamente.")
    except Exception as e:
        st.error(f"Error al interpretar la función: {e}")
        funcion = None


a = st.number_input("Límite inferior (a)", value=float(a_def))
b = st.number_input("Límite superior (b)", value=float(b_def))
precision = st.number_input("Precisión (Δx)", value=0.01, min_value=0.0001, step=0.001, format="%.4f")

if b > a:
    n = int(((b - a) / precision)+1)
    if st.button("Ejecutar búsqueda"):
        resultado = busqueda_exhaustiva_animada(a, b, n, funcion)
        st.success(f"Intervalo con mínimo encontrado: {resultado}")
else:
    st.error("El límite superior debe ser mayor que el inferior.")
