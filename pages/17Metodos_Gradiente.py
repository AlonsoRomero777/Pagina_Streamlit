import streamlit as st

st.markdown("<h1 style='text-align: center;'>Métodos de Optimización Basados en Gradiente</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: justify'>

Los métodos de optimización basados en gradiente son una clase fundamental de algoritmos utilizados para encontrar el mínimo (o máximo) de una función. Estos métodos aprovechan la información de la **derivada** (o gradiente en funciones multivariables) para decidir en qué dirección avanzar con el fin de mejorar el valor de la función objetivo.

A diferencia de los métodos directos, que no requieren derivadas y suelen evaluar la función muchas veces para encontrar el mínimo, los métodos basados en gradiente utilizan la pendiente de la función para avanzar de forma más inteligente hacia un óptimo. Esto generalmente los hace **más eficientes y rápidos**, especialmente en problemas de alta dimensión.

### ¿Cómo funcionan?

El principio clave es el siguiente: el **gradiente** de una función en un punto indica la dirección de mayor incremento. Por lo tanto, si queremos minimizar la función, debemos movernos en la **dirección opuesta al gradiente**.

Uno de los algoritmos más conocidos es el **descenso del gradiente**, donde se actualan los valores de las variables siguiendo esta regla:

<center>𝑥<sub>n+1</sub> = 𝑥<sub>n</sub> − α · ∇f(𝑥<sub>n</sub>)</center>

Donde:

- 𝑥<sub>n</sub> es el valor actual del punto,
- α es la **tasa de aprendizaje** o paso de actualización,
- ∇f(𝑥<sub>n</sub>) es el **gradiente de la función** en el punto actual.

### Ventajas

- Son **rápidos** cuando la función es suave y diferenciable.
- Muy útiles en problemas de gran escala, como en el entrenamiento de redes neuronales.
- Explotan la geometría de la función para hacer pasos más informados.

### Limitaciones

- Requieren que la función sea **diferenciable**.
- No funcionan bien en funciones **discretas o con muchas discontinuidades**.
- Pueden **quedarse atrapados en mínimos locales** si la función no es convexa.
- Sensibles a la **elección de la tasa de aprendizaje**.

### Aplicaciones típicas

- Optimización en ingeniería.
- Machine Learning y Deep Learning.
- Ajuste de parámetros en modelos físicos y estadísticos.
- Diseño óptimo de sistemas.

### ¿Qué pasa si no hay derivadas?

En algunos casos, no se cuenta con la derivada analítica de la función. Aun así, es posible usar **derivadas numéricas** (como diferencias finitas) para aproximar el gradiente. Esto permite aplicar estos métodos incluso en contextos donde la función no está expresada de forma explícita.

</div>
""", unsafe_allow_html=True)
