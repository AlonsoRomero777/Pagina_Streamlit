import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time

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
    "Lata (área)": lata,
    "Caja (volumen negativo)": caja,
    "x² + 54/x": funcion_0,
    "x³ + 2x - 3": funcion_1,
    "x⁴ + x² - 33": funcion_2,
    "3x⁴ - 8x³ - 6x² + 12x": funcion_3
}

def golden_section_search(a: float, b: float, epsilon: float, func: callable):
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi

    c = a + resphi * (b - a)
    d = b - resphi * (b - a)
    fc = func(c)
    fd = func(d)

    points = [(c, fc), (d, fd)]
    min_intervalo = (a, b)

    grafico_placeholder = st.empty()
    rango_grafica = (a, b)

    while abs(b - a) > epsilon:
        if fc < fd:
            b, d, fd = d, c, fc
            c = a + resphi * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = b - resphi * (b - a)
            fd = func(d)

        points.append((c, fc))
        points.append((d, fd))
        min_intervalo = (a, b)

        fig = plot_function_with_points(func, rango_grafica[0], rango_grafica[1], points, f"Búsqueda Dorada - {opcion}")
        grafico_placeholder.pyplot(fig)
        time.sleep(0.5) 

    return (a + b) / 2, points, min_intervalo

def plot_function_with_points(func, a, b, points, title):
    x = np.linspace(a, b, 1000)
    y = [func(xi) for xi in x]
    fig, ax = plt.subplots()
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

st.title("Método de Búsqueda Dorada 🔎")

texto_justificado = """
<div style="text-align: justify;">
<p>El <strong>método de búsqueda dorada</strong> (en inglés, <i>Golden Section Search</i>) es una técnica de optimización unidimensional 
utilizada para encontrar el mínimo o máximo de una función unimodal en un intervalo cerrado [a, b], sin necesidad de derivadas.
Es una mejora del método de intervalos por la mitad, ya que reduce el número de evaluaciones
necesarias por iteración mediante una estrategia más eficiente basada en la proporción áurea.</p>

<p>La búsqueda dorada se basa en el hecho de que si una función es unimodal en un intervalo, entonces se puede descartar 
parte del intervalo después de evaluar la función en dos puntos internos específicos. La elección óptima de esos puntos
se hace de manera que el subintervalo resultante mantenga la misma proporción entre los segmentos del nuevo intervalo.</p>

<p>Esta proporción óptima se conoce como la <strong>razón áurea</strong>, aproximadamente igual a 0.618. Los dos puntos internos 
se calculan de forma que no se requiera reevaluar todos los puntos en cada iteración, haciendo el proceso más eficiente 
en términos computacionales.</p>

<p>En cada paso, se evalúa la función en dos puntos internos del intervalo actual. Dependiendo de cuál de los dos valores 
sea menor (o mayor, en caso de maximización), se elimina una parte del intervalo que no contiene el óptimo. 
La reutilización de una de las evaluaciones anteriores evita realizar evaluaciones innecesarias.</p>

<p><strong>Ventajas:👍</strong></p>
<ul>
<li>No requiere derivadas ni información sobre la forma de la función, solo que sea unimodal.</li>
<li>Menor número de evaluaciones que otros métodos básicos como los intervalos por la mitad.</li>
<li>Rápida convergencia al mínimo o máximo dentro del intervalo.</li>
</ul>

<p><strong>Desventajas:👎</strong></p>
<ul>
<li>Menos eficiente que métodos que usan información del gradiente cuando este está disponible.</li>
<li>Requiere que la función sea unimodal en el intervalo inicial, lo cual debe garantizarse previamente.</li>
</ul>

<p>Este método es ideal para problemas de optimización unidimensional donde no se dispone de derivadas o donde las 
evaluaciones de la función son costosas y se busca eficiencia.</p>
</div>
"""

st.markdown(texto_justificado, unsafe_allow_html=True)


opcion = st.selectbox("**Selecciona la función a optimizar**", list(funciones.keys()))
funcion = funciones[opcion]


a = st.number_input("Límite inferior (a)", value=0.0, key="a_dorada")
b = st.number_input("Límite superior (b)", value=5.0, key="b_dorada")
epsilon = st.number_input("Precisión deseada (ε)", value=0.01, min_value=1e-6, format="%.6f")

if st.button("Ejecutar método"):
    if a >= b:
        st.error("El límite inferior debe ser menor que el superior.")
    else:
        x_min, puntos, intervalo_final = golden_section_search(a, b, epsilon, funcion)
        st.success(f"Mínimo aproximado en x ≈ {x_min:.6f}")
        st.write(f"Último intervalo: [{intervalo_final[0]:.6f}, {intervalo_final[1]:.6f}]")

