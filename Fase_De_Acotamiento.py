import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st



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


def fase_acotamiento(x0, delta, lambda_, funcion, max_iter=1000, max_x=1e6):
    x1 = x0
    x2 = x1 + delta if funcion(x1 + delta) < funcion(x1) else x1 - delta
    k = 1
    
    valores_x = [x1, x2]
    valores_y = [funcion(x1), funcion(x2)]
    
    while funcion(x2) < funcion(x1):
        if abs(x2) > max_x or k > max_iter:
            st.warning("Se alcanzó el límite de iteraciones o crecimiento.")
            break
        
        x1 = x2
        delta *= lambda_
        x2 = x1 + delta if x2 > x0 else x1 - delta
        valores_x.append(x2)
        valores_y.append(funcion(x2))
        k += 1

    fig, ax = plt.subplots()
    ax.plot(valores_x, valores_y, 'bo-', label='Evaluaciones')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Fase de Acotamiento')
    ax.legend()


    return min(x1, x2), max(x1, x2)

st.title("Fase de acotamiento 🤖")
st.markdown("""
El método de fase de acotamiento se utiliza para acotar el mínimo de una función. 
Este método garantiza acotar el mínimo de una función unimodal. 
El algoritmo comienza con una suposición inicial y luego encuentra una dirección de
búsqueda basada en dos evaluaciones más de la función en las proximidades de la suposición inicial.
Posteriormente, se adopta una estrategia de búsqueda exponencial para alcanzar el óptimo.\n 
Puedes ajustar el punto inicial, la delta, lambda y elegir la función a optimizar.
""")

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
