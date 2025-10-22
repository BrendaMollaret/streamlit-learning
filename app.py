import streamlit as st
import pandas as pd
import altair as alt

# Título y descripción
st.title("Explorador de Datos Inmobiliarios")
st.write("Visualización interactiva del dataset usado en el modelado.")

# Cargar datos
df = pd.read_csv("data/dataset_limpio.csv")

# Filtros (widgets)
ciudad = st.selectbox("Seleccionar ciudad", sorted(df["ciudad"].unique()))
filtrado = df[df["ciudad"] == ciudad]

# Gráfico Altair
chart = (
    alt.Chart(filtrado)
    .mark_circle(size=60)
    .encode(
        x="m2_total",
        y="precio",
        color="ambientes:N",
        tooltip=["ciudad", "m2_total", "ambientes", "precio"]

    )
    .interactive()
)

st.altair_chart(chart, use_container_width=True)