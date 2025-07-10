

import streamlit as st
st.markdown("""
<style>
footer {
    text-align: center;
    font-size: 14px;
    color: #95a5a6;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align:center;">🚀 Proyecto Final: Algoritmos de Optimización</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align:center; margin-top:-10px;">👨‍💻 Autor: Yahir Alonso Romero Torres</h3>', unsafe_allow_html=True)
st.markdown('---')

st.subheader("Bienvenido, aquí encontraras muchos algoritmos de optimización para tí. 🧠💻")
st.write("""
Esta página contiene una serie de algoritmos que fueron desarrollados a lo largo del curso de Optimización.
Aquí podrás explorar cómo funciona cada método y visualizar su comportamiento paso a paso.

Los algoritmos están organizados por separado en esta plataforma para que puedas consultarlos fácilmente.
""")

st.markdown("## 📚 ¿Qué es la optimización?")
st.write("""
La optimización es una rama de las matemáticas aplicada en la inteligencia artificial, la ingeniería y otras ciencias,
que se enfoca en encontrar la mejor solución posible (mínima o máxima) a un problema bajo ciertas restricciones.

En este proyecto exploraremos algunos algoritmos clásicos como:
- Búsqueda Exhaustiva 🔍
- Fase de Acotamiento 📏
- Métodos Directos
- Entre otros...
""")



col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("💡 Intuitivo")
    st.write("Interfaz simple y clara para visualizar cada paso del algoritmo.")

with col2:
    st.subheader("⚙️ Interactivo")
    st.write("Puedes cambiar los parámetros y ver los resultados en tiempo real.")

with col3:
    st.subheader("📊 Visual")
    st.write("Gráficos dinámicos para entender el comportamiento de los métodos.")

st.markdown("""
<footer>
© 2025 Yahir Alonso Romero Torres · Ingenieria en Inteligencia Artificial · Universidad De Xalapa
</footer>
""", unsafe_allow_html=True)