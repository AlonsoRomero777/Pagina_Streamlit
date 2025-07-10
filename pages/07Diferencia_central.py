import streamlit as st
import numpy as np


def f(x):
    x1, x2 = x[0], x[1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


def gradiente_central(f, x, delta=1e-2):
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = delta
        grad[i] = (f(x + dx) - f(x - dx)) / (2 * delta)
    return grad


def hessiana_central(f, x, delta=1e-2):
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx_i = np.zeros(n)
            dx_j = np.zeros(n)
            dx_i[i] = delta
            dx_j[j] = delta
            fpp = f(x + dx_i + dx_j)
            fpm = f(x + dx_i - dx_j)
            fmp = f(x - dx_i + dx_j)
            fmm = f(x - dx_i - dx_j)
            hess[i, j] = (fpp - fpm - fmp + fmm) / (4 * delta**2)
    return hess


st.title("Gradiente y Hessiana - Función de Himmelblau")

texto_justificado = """
<div style="text-align: justify;">
El método de la diferencia central es una técnica numérica utilizada para estimar derivadas de funciones,
especialmente útil cuando no se dispone de una expresión analítica de la derivada o cuando se trabaja con 
datos discretos.

Este método pertenece a la familia de las diferencias finitas, que son aproximaciones basadas en valores de 
la función evaluada en puntos cercanos al de interés. A diferencia de las aproximaciones por diferencia hacia adelante o hacia atrás, 
la diferencia central utiliza valores a ambos lados del punto en cuestión, lo que la hace más precisa. Matemáticamente, 
la primera derivada de una función f(x) se estima como:
</div>
"""
st.markdown(texto_justificado, unsafe_allow_html=True)

st.latex(r"f'(x) \approx \frac{f(x + \Delta x) - f(x - \Delta x)}{2 \Delta x}")

texto_justifi = """
<div style="text-align: justify;">
Una ventaja importante del método de diferencia central es su simetría,
lo que minimiza el sesgo que puede introducirse al usar sólo un lado del punto.
Sin embargo, también requiere evaluar la función en dos puntos por cada derivada,
lo que puede ser costoso si la evaluación es compleja o si se aplica muchas veces.
</div>
"""
st.markdown(texto_justifi, unsafe_allow_html=True)

x1 = st.number_input("**Valor de x₁**:", value=1.0, step=0.1)
x2 = st.number_input("**Valor de x₂**:", value=1.0, step=0.1)
delta = st.number_input("**Valor de Δ (diferencia central)**:", min_value=0.0001, value=0.01, step=0.001, format="%.4f")

x_eval = np.array([x1, x2])

if st.button("Calcular Gradiente y Hessiana"):
    grad = gradiente_central(f, x_eval, delta)
    hess = hessiana_central(f, x_eval, delta)

    st.subheader("Gradiente")
    st.write(f"∇f({x1:.2f}, {x2:.2f}) = {grad}")

    st.subheader("Matriz Hessiana")
    st.write(hess)
