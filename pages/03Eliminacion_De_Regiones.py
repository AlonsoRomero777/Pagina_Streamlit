import streamlit as st
from PIL import Image

st.title("Metodos de eliminacion de regiones")

texto_justificado = """
<div style="text-align: justify;">
Los "métodos de eliminación de regiones" se utilizan en optimización unidimensional para reducir progresivamente el intervalo de búsqueda donde se encuentra el mínimo de una función.

El principio básico consiste en evaluar la función objetivo en dos puntos dentro del intervalo y, dependiendo del resultado, **descartar regiones donde el mínimo no puede estar**.

📌 Reglas fundamentales:

Supongamos que tenemos una función continua \( f(x) \) en un intervalo \( (a, b) \), y evaluamos la función en dos puntos internos \( x_1 \) y \( x_2 \), con \( a < x_1 < x_2 < b \). Entonces:

- Si \( f(X1) > f(X2) \), entonces el mínimo **no se encuentra en** \( (a, X1) \).
- Si \( f(X1) < f(X2) \), entonces el mínimo **no se encuentra en** \( (X2, b) \).
- Si \( f(X1) = f(X2) \), entonces el mínimo **no se encuentra en** \( (a, X1) \) **ni en** \( (X2, b) \).
</div>
"""

st.markdown(texto_justificado, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])  # El centro es col2

with col2:
    image = Image.open("Curva.png")
    st.image(image, caption="", width=600)