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

import time
def fase_acotamiento(x0, delta, lambda_, funcion, max_iter=1000, max_x=1e6):
    x1 = x0
    x2 = x1 + delta if funcion(x1 + delta) < funcion(x1) else x1 - delta
    k = 1
    
    valores_x = [x1, x2]
    valores_y = [funcion(x1), funcion(x2)]
    
    grafico_placeholder = st.empty()
    
    while funcion(x2) < funcion(x1):
        if abs(x2) > max_x or k > max_iter:
            st.warning("Se alcanzó el límite de iteraciones o crecimiento.")
            break
        
        # Visualizar progreso en cada paso
        fig, ax = plt.subplots()
        ax.plot(valores_x, valores_y, 'bo-', label='Evaluaciones')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Iteración {k}: Fase de Acotamiento')
        ax.legend()
        grafico_placeholder.pyplot(fig)
        time.sleep(0.4)

        # Continuar con el algoritmo
        x1 = x2
        delta *= lambda_
        x2 = x1 + delta if x2 > x0 else x1 - delta
        valores_x.append(x2)
        valores_y.append(funcion(x2))
        k += 1

    # Mostrar gráfico final completo
    fig, ax = plt.subplots()
    ax.plot(valores_x, valores_y, 'ro-', label='Evaluaciones Finales')
    ax.axvline(min(x1, x2), color='green', linestyle='--', label='Límite inferior')
    ax.axvline(max(x1, x2), color='blue', linestyle='--', label='Límite superior')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Fase de Acotamiento: Intervalo Final')
    ax.legend()
    grafico_placeholder.pyplot(fig)

    return min(x1, x2), max(x1, x2)


st.title("Fase de Acotamiento 🤖")

st.markdown("""
<style>
.justify-text {
    text-align: justify;
}
</style>

<div class="justify-text">

<p>El <strong>método de fase de acotamiento</strong> es una técnica utilizada para <strong>localizar un intervalo</strong> que contenga 
el mínimo de una función unimodal. Se trata de una fase preliminar dentro de muchos algoritmos de optimización unidimensional, 
ya que no busca directamente el mínimo, sino que lo acota dentro de un rango definido para posteriormente aplicar métodos más precisos.</p>

<p>El algoritmo comienza con un <strong>punto inicial</strong> y una <strong>delta</strong> (pequeño desplazamiento), y evalúa la función en dos 
direcciones: una hacia adelante y otra hacia atrás. En función de las evaluaciones, determina hacia dónde moverse para continuar con la búsqueda. 
Posteriormente, se aplica una estrategia de búsqueda <strong>exponencial</strong> utilizando un parámetro de expansión <strong>λ (lambda)</strong>, 
aumentando progresivamente la distancia hasta que se detecta un cambio de tendencia en los valores de la función.</p>

<p>Este proceso garantiza, bajo el supuesto de <strong>unimodalidad</strong> de la función, que el mínimo se encuentra dentro del intervalo final 
generado por el método.</p>

<h4>👍 Ventajas:</h4>
<ul>
<li>No requiere derivadas.</li>
<li>Simple de implementar.</li>
<li>Sirve como etapa previa para otros métodos (como búsqueda dorada o parabolización).</li>
</ul>

<h4>👎 Desventajas:</h4>
<ul>
<li>No encuentra el mínimo exacto, solo lo acota.</li>
<li>Puede ser ineficiente si la función no es unimodal en el intervalo.</li>
<li>Sensibilidad a los parámetros iniciales (delta, lambda).</li>
</ul>

<h4> ¿Qué puedes hacer en esta aplicación?</h4>
<ul>
<li>Seleccionar la función a optimizar.</li>
<li>Definir el <strong>punto inicial</strong> de la búsqueda.</li>
<li>Ajustar el tamaño de paso <strong>Δ (delta)</strong>.</li>
<li>Configurar el factor de expansión <strong>λ (lambda)</strong>.</li>
<li>Visualizar el proceso de expansión y el intervalo resultante donde se encuentra el mínimo.</li>
</ul>

</div>
""", unsafe_allow_html=True)

funciones = {
    "x² + 3": funcion_00,
    "Lata (área)": lata,
    "Caja (volumen negativo)": caja,
    "x² + 54/x": funcion_0,
    "x³ + 2x - 3": funcion_1,
    "x⁴ + x² - 33": funcion_2,
    "3x⁴ - 8x³ - 6x² + 12x": funcion_3,
    "Función Personalizada": None
}

opcion = st.selectbox("**Selecciona la función**", list(funciones.keys()))

if opcion == "Función Personalizada":
    expresion = st.text_input("Escribe tu función en términos de x (ej. x**2 + 3*x - 1):", value="x**2")
    try:
        funcion_seleccionada = lambda x: eval(expresion, {"x": x, "math": math, "np": np})
        st.success("Función personalizada cargada correctamente.")
    except Exception as e:
        st.error(f"Error al interpretar la función: {e}")
        funcion_seleccionada = None
else:
    funcion_seleccionada = funciones[opcion]

x0 = st.number_input("Punto inicial (x₀)", value=1.0)
delta = st.number_input("Delta inicial (Δ)", value=0.01, min_value=0.0001)
lambda_ = st.number_input("Lambda (λ)", value=2.0, min_value=1.01)


if st.button("Ejecutar fase de acotamiento"):
    intervalo = fase_acotamiento(x0, delta, lambda_, funcion_seleccionada)
    st.success(f"Intervalo encontrado: [{intervalo[0]:.4f}, {intervalo[1]:.4f}]")
