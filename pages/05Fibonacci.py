import numpy as np
import matplotlib.pyplot as plt
import math
import streamlit as st
import time

# Funciones objetivo
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

def fibonacci(n):
    fib = [0, 1]
    for _ in range(2, n+1):
        fib.append(fib[-1] + fib[-2])
    return fib

def Busqueda_Fibonacci(a: float, b: float, n: int, epsilon: float, func: callable):
    fib = fibonacci(n+1)
    points = []

    k = 2
    L = b - a

    Lk = (fib[n-k+1] / fib[n+1]) * L
    x1 = a + Lk
    x2 = b - Lk

    f_x1 = func(x1)
    f_x2 = func(x2)

    points.append((x1, f_x1))
    points.append((x2, f_x2))
    min_intervalo = (a, b)

    grafico_placeholder = st.empty()
    rango_grafica = (a, b)
    
    while k < n:
        if f_x1 > f_x2:
            a = x1
            x1 = x2
            x2 = b - ((fib[n-k] / fib[n+1]) * L)
            f_x1 = f_x2
            f_x2 = func(x2)
            points.append((x2, f_x2))
        else:
            b = x2
            x2 = x1
            x1 = a + ((fib[n-k] / fib[n+1]) * L)
            f_x2 = f_x1
            f_x1 = func(x1)
            points.append((x1, f_x1))

        min_intervalo = (a, b)
        k += 1

        fig = plot_function_with_points(func, rango_grafica[0], rango_grafica[1], points, f"Búsqueda Fibonacci - {opcion}")
        grafico_placeholder.pyplot(fig)
        time.sleep(0.5)  

    return (a + b) / 2, points, min_intervalo


def plot_function_with_points(func, a, b, points, title):
    x = np.linspace(a, b, 1000)
    y = [func(xi) for xi in x]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y, label='Función', color='blue')
    if points:
        x_points, y_points = zip(*points)
        ax.scatter(x_points, y_points, color='red', label='Puntos evaluados')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend()
    return fig

st.title("Búsqueda Fibonacci 🕵️‍♂️")

texto_justificado = """
<div style="text-align: justify;">
<p>El <strong>método de Fibonacci</strong> es una técnica de optimización unidimensional que, al igual que la búsqueda dorada,
se utiliza para encontrar el mínimo o máximo de una función unimodal en un intervalo cerrado 
[a, b], sin requerir el uso de derivadas. Sin embargo, el método de Fibonacci puede ser incluso más
eficiente que la búsqueda dorada cuando se conoce de antemano el número de iteraciones que se 
desean realizar. Esta eficiencia se logra mediante el uso de los números de Fibonacci para determinar 
los puntos de evaluación dentro del intervalo.</p>

<p>La base del método es muy similar a la búsqueda dorada: 
se eligen dos puntos dentro del intervalo, se evalúa la función en ellos y, 
dependiendo de cuál valor sea menor, se reduce el intervalo. Lo que lo distingue es 
que las posiciones de los puntos dentro del intervalo no se eligen según la razón áurea, 
sino según la proporción de números de Fibonacci consecutivos, que van cambiando en cada iteración.</p>

<p>El algoritmo requiere calcular previamente los números de Fibonacci hasta alcanzar una cantidad 
relacionada con la precisión deseada (es decir, el número de iteraciones), y utiliza esos valores 
para ubicar los puntos de evaluación. A medida que se avanza en el proceso, el intervalo se 
estrecha de forma controlada, garantizando una buena aproximación al mínimo (o máximo) de la función.</p>

<p><strong>Ventajas:👍</strong></p>
<ul>
<li>No requiere derivadas, por lo que es adecuado para funciones no diferenciables.</li>
<li>Es más eficiente que otros métodos de búsqueda unidimensional cuando se fija un número de iteraciones.</li>
<li>Reduce el número de evaluaciones de la función al mínimo teórico posible en ese marco.</li>
</ul>

<p><strong>Desventajas:👎</strong></p>
<ul>
<li>Requiere conocer de antemano el número total de evaluaciones (o iteraciones).</li>
<li>No es adaptativo; si se necesita más precisión, se debe reiniciar el proceso con nuevos valores de Fibonacci.</li>
</ul>

<p>En general, la búsqueda Fibonacci es una excelente opción para optimización unidimensional cuando se puede
predecir el número de pasos, y proporciona resultados similares a la búsqueda dorada, pero con menor
número de evaluaciones.</p>
</div>
"""

st.markdown(texto_justificado, unsafe_allow_html=True)


opcion = st.selectbox("**Selecciona la función a optimizar**", list(funciones.keys()))
funcion = funciones[opcion]

a = st.number_input("Límite inferior (a)", value=0.0, key="a_fib")
b = st.number_input("Límite superior (b)", value=5.0, key="b_fib")
n = st.number_input("Número de evaluaciones (n ≥ 3)", min_value=3, step=1, value=10, key="n_fib")
epsilon = st.number_input("Precisión deseada (ε)", value=0.0001, min_value=1e-6, format="%.6f")

if st.button("Ejecutar método"):
    if a >= b:
        st.error("El límite inferior debe ser menor que el superior.")
    else:
        x_min, puntos, intervalo_final = Busqueda_Fibonacci(a, b, n, epsilon, funcion)
        st.success(f"Mínimo aproximado en x ≈ {x_min:.6f}")
        st.write(f"Último intervalo: [{intervalo_final[0]:.6f}, {intervalo_final[1]:.6f}]")

