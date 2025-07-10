import streamlit as st
import time

st.set_page_config(
    page_title="Algoritmos de Optimización",
    layout='wide',
    menu_items={'About': "Proyecto final de Optimización. Autor: Yahir Alonso Romero Torres"}
)

if "carga_completa" not in st.session_state:
    with st.spinner("🔄 CARGANDO PÁGINA..."):
        time.sleep(2.5)
    st.session_state.carga_completa = True


teoria = st.Page("main.py", title="Teoría", icon=":material/captive_portal:", default=True)

exhaustive_search = st.Page("pages/01Busqueda_exhaustiva.py", title="Búsqueda Exhaustiva", icon=":material/open_in_full:")
bounding_phase = st.Page("pages/02Fase_acotamiento.py", title="Fase De Acotamiento", icon=":material/open_in_full:")

Reglas_eliminacion = st.Page("pages/03Eliminacion_De_Regiones.py", title="Eliminacion de regiones", icon=":material/search_check:")
interval_halving = st.Page("pages/04Intervalos_por_mitad.py", title="Intervalos por la Mitad", icon=":material/search_check:")
fibonacci_search = st.Page("pages/05Fibonacci.py", title="Fibonacci", icon=":material/search_check:")
golden_section = st.Page("pages/06Busqueda_dorada.py", title="Búsqueda Dorada", icon=":material/search_check:")

centralDifference = st.Page("pages/07Diferencia_central.py", title="Método de la Diferencia Central", icon=":material/bottom_right_click:")
newton_raphson = st.Page("pages/08Newton_Raphson.py", title="Método de Newton-Raphson", icon=":material/bottom_right_click:")
Biseccion = st.Page("pages/09Biseccion.py", title="Método de Bisección", icon=":material/bottom_right_click:")
secante = st.Page("pages/10Secante.py", title="Método de la Secante", icon=":material/bottom_right_click:")

busqUnidireccional = st.Page("pages/11Busqueda_Unidireccional.py", title="Búsqueda Unidireccional", icon=":material/hive:")
nelderMead = st.Page("pages/12Nelder_Mead.py", title="Nelder Mead (Simplex)", icon=":material/hive:")
hookeJeeves = st.Page("pages/13Hooke_Jeeves.py", title="Hooke Jeeves (Búsqueda de Patrón)", icon=":material/hive:")
randomWalk = st.Page("pages/14Random_Walk.py", title="Caminata Aleatoria (Random Walk)", icon=":material/hive:")
hillClimbing = st.Page("pages/15Hill_Climbing.py", title="Ascenso de la Colina (Hill Climbing)", icon=":material/hive:")
simulatedAnnealing = st.Page("pages/16Simulated_Annealing.py", title="Recocido Simulado (Simulated Annealing)", icon=":material/hive:")
gradiente = st.Page("pages/17Metodos_Gradiente.py", title="Métodos basados en gradiente", icon=":material/manage_search:")
cauchy = st.Page("pages/18Cauchy.py", title="Método de Cauchy", icon=":material/manage_search:")
newton = st.Page("pages/19Metodo_Newton.py", title="Método de Newton", icon=":material/manage_search:")

# Navegación
pg = st.navigation(
    {
        "OPTIMIZACIÓN": [teoria],
        "MÉTODOS DE ACOTAMIENTO": [exhaustive_search, bounding_phase],
        "MÉTODOS DE ELIMINACION DE REGIONES": [interval_halving, fibonacci_search, golden_section, Reglas_eliminacion],
        "MÉTODO BASADO EN GRADIENTE": [centralDifference, newton_raphson, Biseccion, secante],
        "ALGORITMOS DE OPTIMIZACIÓN MULTIVARIADA": [busqUnidireccional, nelderMead, hookeJeeves, randomWalk, hillClimbing, simulatedAnnealing],
        "MÉTODOS BASADOS EN GRADIENTE": [gradiente, cauchy, newton]
    }
)
st.markdown("""
<style>
/* Cambia el color de fondo de la barra lateral */
section[data-testid="stSidebar"] {
    background-color: #7e0e10; 
    color: white;
}

/* Cambia el color del texto (etiquetas, opciones) */
section[data-testid="stSidebar"] * {
    color: white;
}

/* Cambia el color de los títulos de secciones ("Optimización", "Métodos de Acotamiento", etc.) */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #00ffff; 
}

/* Cambia el color de fondo cuando se pasa el mouse sobre una opción */
section[data-testid="stSidebar"] a:hover {
    background-color: #c6b14c;
}
</style>
""", unsafe_allow_html=True)

pg.run()
