import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time

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

funciones = {
    "x² + 3": funcion_00,
    "Lata (área)": lata,
    "Caja (volumen negativo)": caja,
    "x² + 54/x": funcion_0,
    "x³ + 2x - 3": funcion_1,
    "x⁴ + x² - 33": funcion_2,
    "3x⁴ - 8x³ - 6x² + 12x": funcion_3
}


def intervalos_mitad(a: float, b: float, epsilon: float, func: callable, rango_grafica=None):
    points = []
    intervalos = []

    if rango_grafica is None:
        rango_grafica = (a, b)  

    grafico_placeholder = st.empty()  

    while (b - a) > epsilon:
        xm = (a + b) / 2
        L = b - a
        x1 = a + L / 4
        x2 = b - L / 4
        f_xm = func(xm)
        f_x1 = func(x1)
        f_x2 = func(x2)

        points.append((x1, f_x1))
        points.append((x2, f_x2))
        intervalos.append((a, b))

        fig = plot_function_with_points(func, rango_grafica[0], rango_grafica[1], points, f"Intervalos por la Mitad - {opcion}")
        grafico_placeholder.pyplot(fig)
        time.sleep(0.5) 

        if f_x1 < f_xm:
            b = xm
        elif f_x2 < f_xm:
            a = xm
        else:
            a, b = x1, x2

    return (a + b) / 2, points, intervalos



def plot_function_with_points(func, a, b, points, title):
    x = np.linspace(a, b, 1000)
    y = [func(xi) for xi in x]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y, label='Función', color='blue')

    if points:
        x_points, y_points = zip(*points)
        ax.scatter(x_points, y_points, color='red', label='Puntos evaluados')

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True)
    ax.legend()
    
    return fig


st.title("Intervalos por la Mitad")

texto_justificado = """
<div style="text-align: justify;">
<p>El método de <strong>intervalos por la mitad</strong>, también conocido como <i>Interval Halving Method</i>, 
es una técnica de optimización que se utiliza para encontrar el mínimo (o en ocasiones el máximo) 
de una función en un intervalo cerrado, siempre que dicha función sea unimodal en ese intervalo.</p>

<p>Una función unimodal es aquella que presenta un solo mínimo (o máximo) dentro del intervalo analizado. 
Este método pertenece a la categoría de técnicas de <strong>búsqueda directa</strong>, lo que significa que no necesita 
derivadas ni información sobre la pendiente de la función. Es un método determinista y sencillo que reduce 
el intervalo de búsqueda en cada iteración, lo que facilita su implementación.</p>

<p>El procedimiento consiste en dividir el intervalo actual en cuatro partes iguales, evaluando la función 
en tres puntos estratégicos: el punto medio del intervalo y dos puntos simétricos a una distancia fija a la izquierda 
y derecha del centro. Dependiendo de los valores obtenidos, se elimina la sección que no contiene el mínimo 
y se continúa el proceso con el nuevo intervalo más pequeño.</p>

<p><strong>Ventajas: 👍</strong></p>
<ul>
<li>Fácil de implementar y comprender.</li>
<li>No requiere derivadas ni conocimientos del comportamiento de la función más allá de su unimodalidad.</li>
<li>Reduce de forma constante el intervalo de búsqueda, garantizando convergencia al óptimo.</li>
</ul>

<p><strong>Desventajas:👎</strong></p>
<ul>
<li>Puede requerir más evaluaciones que otros métodos más eficientes como la búsqueda dorada o Fibonacci.</li>
<li>No se adapta automáticamente a funciones con comportamiento complejo fuera del supuesto unimodal.</li>
</ul>

<p>Este método es ideal para situaciones donde se necesita una solución rápida, robusta y sin requerimientos avanzados, 
especialmente útil en problemas de optimización unidimensional con funciones costosas de evaluar o no derivables.</p>
</div>
"""

st.markdown(texto_justificado, unsafe_allow_html=True)


opcion = st.selectbox("**Selecciona la función a optimizar**", list(funciones.keys()))
funcion = funciones[opcion]

a = st.number_input("Límite inferior (a)", value=0.0, key="a_mitad")
b = st.number_input("Límite superior (b)", value=5.0, key="b_mitad")
epsilon = st.number_input("Precisión deseada (ε)", value=0.0001, min_value=1e-6, format="%.6f")

if st.button("Ejecutar método"):
    if a >= b:
        st.error("El límite inferior debe ser menor que el superior.")
    else:
        x_min, puntos, intervalos = intervalos_mitad(a, b, epsilon, funcion, rango_grafica=(a, b))
        st.success(f"Mínimo aproximado en x ≈ {x_min:.6f}")
        st.write(f"Último intervalo: [{intervalos[-1][0]:.6f}, {intervalos[-1][1]:.6f}]")

        

