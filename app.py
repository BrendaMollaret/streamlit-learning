import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO
import os
from math import radians, sin, cos, asin, sqrt
import base64
from src.process_user_input import process_user_input, UserInput, GeoPoint

st.title("Buscador de Propiedades Inteligente")

# Lista simple de stopwords en español (sklearn nativamente solo soporta 'english')
SPANISH_STOPWORDS = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un",
    "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le",
    "ya", "o", "fue", "este", "ha", "sí", "porque", "esta", "son", "entre", "cuando",
    "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien", "desde",
    "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese",
    "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo",
    "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada",
    "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros",
    "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros",
    "vosotras", "os", "mío", "mía", "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas",
    "suyo", "suya", "suyos", "suyas", "nuestro", "nuestra", "nuestros", "nuestras",
    "vuestro", "vuestra", "vuestros", "vuestras", "esos", "esas"
]

# --- Utilidades de imagen: descarga y recorte a relación 8:5 (400x250) ---
TARGET_ASPECT = 8 / 5
TARGET_MIN_HEIGHT = 320  # px; controla tamaño inicial visible

@st.cache_data(show_spinner=False)
def fetch_image_bytes(url: str) -> bytes | None:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200 and resp.content:
            return resp.content
    except Exception:
        return None
    return None

def center_crop_to_aspect(img: Image.Image, target_aspect: float) -> Image.Image:
    width, height = img.size
    current_aspect = width / height
    if abs(current_aspect - target_aspect) < 1e-3:
        return img
    if current_aspect > target_aspect:
        # Imagen más ancha: recortar lados
        new_width = int(target_aspect * height)
        x1 = (width - new_width) // 2
        x2 = x1 + new_width
        return img.crop((x1, 0, x2, height))
    else:
        # Imagen más alta: recortar arriba/abajo
        new_height = int(width / target_aspect)
        y1 = (height - new_height) // 2
        y2 = y1 + new_height
        return img.crop((0, y1, width, y2))

def prepare_card_image(url: str) -> Image.Image | None:
    content = fetch_image_bytes(url)
    if not content:
        return None
    try:
        img = Image.open(BytesIO(content)).convert("RGB")
        img = center_crop_to_aspect(img, TARGET_ASPECT)
        # Redimensionar a altura mínima para que se vea grande de entrada
        w, h = img.size
        if h < TARGET_MIN_HEIGHT:
            new_h = TARGET_MIN_HEIGHT
            new_w = int(new_h * (w / h))
            img = img.resize((new_w, new_h), Image.LANCZOS)
        return img
    except Exception:
        return None

def render_full_width_image(img: Image.Image | None, fallback_url: str | None = None) -> None:
    """Render a full-width image without lightbox using HTML.

    If a PIL image is provided, it is encoded to base64 and embedded inline.
    If not, and a fallback URL is provided, that URL is used directly.
    """
    try:
        if img is not None:
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
            src = f"data:image/jpeg;base64,{b64}"
        elif isinstance(fallback_url, str) and fallback_url:
            src = fallback_url
        else:
            return
        st.markdown(
            f"<img src='{src}' style='width:100%;height:auto;display:block;border-radius:8px;' />",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

# --- Dataset de ejemplo (incluye coordenadas) ---
data = [
    {
        "ciudad": "Godoy Cruz",
        "ubicacion": "Calle Sarmiento 1200",
        "alquiler": "250.000",
        "m2_total": 120,
        "ambientes": 3,
        "banos": 2,
        "imagen": "https://images.pexels.com/photos/186077/pexels-photo-186077.jpeg?_gl=1*2ciflt*_ga*MTI1MzQyNDY2MC4xNzYyMzUyMDk0*_ga_8JE65Q40S6*czE3NjIzNTIwOTQkbzEkZzEkdDE3NjIzNTIxMDMkajUxJGwwJGgw",
        "descripcion": "Casa moderna con jardín y pileta, excelente iluminación natural.",
        "lat": -32.931, "lon": -68.845
    },
    {
        "ciudad": "Luján de Cuyo",
        "ubicacion": "San Martín 500",
        "alquiler": "180.000",
        "m2_total": 90,
        "ambientes": 2,
        "banos": 1,
        "imagen": "https://images.pexels.com/photos/1396132/pexels-photo-1396132.jpeg?_gl=1*1dd7gx5*_ga*MTI1MzQyNDY2MC4xNzYyMzUyMDk0*_ga_8JE65Q40S6*czE3NjIzNTIwOTQkbzEkZzEkdDE3NjIzNTIxODQkajU1JGwwJGgw",
        "descripcion": "Departamento céntrico, sin jardín, ideal para pareja o estudiante.",
        "lat": -33.041, "lon": -68.878
    },
    {
        "ciudad": "Maipú",
        "ubicacion": "Ruta 60 km 10",
        "alquiler": "270.000",
        "m2_total": 150,
        "ambientes": 4,
        "banos": 2,
        "imagen": "https://images.pexels.com/photos/1396122/pexels-photo-1396122.jpeg?_gl=1*19p2osr*_ga*MTI1MzQyNDY2MC4xNzYyMzUyMDk0*_ga_8JE65Q40S6*czE3NjIzNTIwOTQkbzEkZzEkdDE3NjIzNTIyMDgkajMxJGwwJGgw",
        "descripcion": "Casa amplia con patio, quincho y espacio verde.",
        "lat": -32.985, "lon": -68.792
    },
    {
        "ciudad": "Ciudad de Mendoza",
        "ubicacion": "Av. Colón 800",
        "alquiler": "320.000",
        "m2_total": 85,
        "ambientes": 2,
        "banos": 1,
        "imagen": "https://images.pexels.com/photos/7710011/pexels-photo-7710011.jpeg?_gl=1*pktlag*_ga*MTI1MzQyNDY2MC4xNzYyMzUyMDk0*_ga_8JE65Q40S6*czE3NjIzNTIwOTQkbzEkZzEkdDE3NjIzNTIyNjckajU5JGwwJGgw",
        "descripcion": "Departamento de lujo con vista a la montaña, sin patio.",
        "lat": -32.889, "lon": -68.845
    },
    {
        "ciudad": "Guaymallén",
        "ubicacion": "Paso de los Andes 200",
        "alquiler": "220.000",
        "m2_total": 110,
        "ambientes": 3,
        "banos": 2,
        "imagen": "https://images.pexels.com/photos/33043467/pexels-photo-33043467.jpeg?_gl=1*16csimg*_ga*MTI1MzQyNDY2MC4xNzYyMzUyMDk0*_ga_8JE65Q40S6*czE3NjIzNTIwOTQkbzEkZzEkdDE3NjIzNTIyODMkajQzJGwwJGgw",
        "descripcion": "Casa cómoda con pequeño jardín, cochera y patio techado.",
        "lat": -32.900, "lon": -68.792
    },
]

df = pd.DataFrame(data)

# --- Vectorización de descripciones ---
vectorizer = TfidfVectorizer(stop_words=SPANISH_STOPWORDS)
embeddings = vectorizer.fit_transform(df["descripcion"])

# --- Modo de búsqueda ---
mood = st.selectbox(
    "Modo de búsqueda",
    ["Texto libre", "Parámetros", "Mapa"],
    index=0
)

def render_card(row: pd.Series, extra_label: str | None = None) -> None:
    with st.container(border=True):
        img = prepare_card_image(row["imagen"]) if isinstance(row["imagen"], str) else None
        render_full_width_image(img, fallback_url=row.get("imagen") if isinstance(row.get("imagen"), str) else None)
        st.markdown(f"### {row['ciudad']} – {row['ubicacion']}")
        if extra_label:
            st.caption(extra_label)
        st.write(f"💰 **Alquiler:** ${row['alquiler']}")
        st.write(f"📏 {row['m2_total']} m² | 🛏 {row['ambientes']} amb | 🚿 {row['banos']} baños")
        st.write(f"📝 {row['descripcion']}")

if mood == "Texto libre":
    query = st.text_input("¿Qué estás buscando? (ej: casa con jardín y pileta en Godoy Cruz)")
    # Forzar modelo fijo sin permitir elección del usuario
    modelo = "sentence_transformer"
    output_qty = st.number_input("Cantidad de resultados", min_value=1, max_value=20, value=5, step=1)

    if st.button("Buscar") and query:
        with st.spinner("Consultando recomendaciones..."):
            try:
                input_data = UserInput(
                    texto=query,
                    modelo=modelo,
                    output_qty=int(output_qty)
                )
                data_out = process_user_input(input_data)
                
                if data_out.get("status") == "error":
                    st.error(f"Error: {data_out.get('message', 'Error desconocido')}")
                else:
                    # Get properties from the output
                    output = data_out.get("output", {})
                    if isinstance(output, dict) and "error" in output:
                        st.error(f"Error: {output.get('error', 'Error desconocido')}")
                    else:
                        props = output.get("properties", []) if isinstance(output, dict) else []
                        if not props:
                            st.info("Sin resultados.")
                        else:
                            st.subheader("Resultados:")
                        for prop in props:
                            row = {
                                "ciudad": prop.get("ciudad") or "",
                                "ubicacion": prop.get("ubicacion") or prop.get("direccion") or "",
                                "alquiler": prop.get("alquiler") if prop.get("alquiler") is not None else "",
                                "m2_total": int(prop.get("m2_total")) if isinstance(prop.get("m2_total"), (int, float)) else prop.get("m2_total"),
                                "ambientes": int(prop.get("ambientes")) if isinstance(prop.get("ambientes"), (int, float)) else prop.get("ambientes"),
                                "banos": int(prop.get("banos")) if isinstance(prop.get("banos"), (int, float)) else prop.get("banos"),
                                "imagen": prop.get("imagen"),
                                "descripcion": prop.get("descripcion") or "",
                            }
                            extra = None
                            if prop.get("score_total") is not None:
                                extra = f"Puntaje: {prop['score_total']:.3f}"
                            elif prop.get("similarity_score") is not None:
                                extra = f"Similitud: {prop['similarity_score']:.3f}"
                            render_card(pd.Series(row), extra_label=extra)
                            if isinstance(prop.get("url"), str) and prop.get("url"):
                                try:
                                    st.link_button("Ver aviso", prop["url"], use_container_width=True)
                                except Exception:
                                    st.write(f"[Ver aviso]({prop['url']})")
            except Exception as e:
                st.error(f"Error al procesar la solicitud: {e}")
                import traceback
                st.code(traceback.format_exc())

elif mood == "Parámetros":
    col1, col2, col3 = st.columns(3)
    with col1:
        ciudades = ["Todas"] + sorted(df["ciudad"].unique().tolist())
        ciudad = st.selectbox("Ciudad", ciudades)
    with col2:
        min_m2 = st.number_input("Mín. m²", min_value=0, value=0, step=10)
        min_amb = st.number_input("Mín. ambientes", min_value=0, value=0, step=1)
    with col3:
        min_banos = st.number_input("Mín. baños", min_value=0, value=0, step=1)
        max_m2 = st.number_input("Máx. m²", min_value=0, value=0, step=10)
        min_cocheras = st.number_input("Mín. cocheras", min_value=0, value=0, step=1)
        min_alq = st.number_input("Alquiler mín.", min_value=0, value=0, step=1000)
        max_alq = st.number_input("Alquiler máx.", min_value=0, value=0, step=1000)

    modelo = "sentence_transformer"
    output_qty = st.number_input("Cantidad de resultados", min_value=1, max_value=20, value=5, step=1)

    if st.button("Filtrar"):
        # Construir un texto de consulta a partir de los parámetros
        partes = []
        # No incluir ciudad en el texto generado
        # Rango de m2
        if min_m2 > 0 and max_m2 > 0:
            if max_m2 >= min_m2:
                partes.append(f"entre {int(min_m2)} y {int(max_m2)} m²")
            else:
                partes.append(f"al menos {int(min_m2)} m²")
        elif min_m2 > 0:
            partes.append(f"al menos {int(min_m2)} m²")
        elif max_m2 > 0:
            partes.append(f"hasta {int(max_m2)} m²")
        # Ambientes y baños
        if min_amb > 0:
            partes.append(f"con al menos {int(min_amb)} ambientes")
        if min_banos > 0:
            partes.append(f"con al menos {int(min_banos)} baños")
        if min_cocheras > 0:
            partes.append(f"con al menos {int(min_cocheras)} cocheras")
        # Alquiler
        if min_alq > 0 and max_alq > 0:
            if max_alq >= min_alq:
                partes.append(f"alquiler entre {int(min_alq)} y {int(max_alq)}")
            else:
                partes.append(f"alquiler desde {int(min_alq)}")
        elif min_alq > 0:
            partes.append(f"alquiler desde {int(min_alq)}")
        elif max_alq > 0:
            partes.append(f"alquiler hasta {int(max_alq)}")

        consulta = " ".join(partes).strip()
        if not consulta:
            consulta = "propiedades"  # fallback minimal

        with st.spinner("Consultando recomendaciones..."):
            try:
                input_data = UserInput(
                    texto=consulta,
                    modelo=modelo,
                    output_qty=int(output_qty)
                )
                data_out = process_user_input(input_data)
                
                if data_out.get("status") == "error":
                    st.error(f"Error: {data_out.get('message', 'Error desconocido')}")
                else:
                    # Get properties from the output
                    output = data_out.get("output", {})
                    if isinstance(output, dict) and "error" in output:
                        st.error(f"Error: {output.get('error', 'Error desconocido')}")
                    else:
                        props = output.get("properties", []) if isinstance(output, dict) else []
                        if not props:
                            st.info("Sin resultados.")
                        else:
                            st.subheader("Resultados:")
                        for prop in props:
                            row = {
                                "ciudad": prop.get("ciudad") or "",
                                "ubicacion": prop.get("ubicacion") or prop.get("direccion") or "",
                                "alquiler": prop.get("alquiler") if prop.get("alquiler") is not None else "",
                                "m2_total": int(prop.get("m2_total")) if isinstance(prop.get("m2_total"), (int, float)) else prop.get("m2_total"),
                                "ambientes": int(prop.get("ambientes")) if isinstance(prop.get("ambientes"), (int, float)) else prop.get("ambientes"),
                                "banos": int(prop.get("banos")) if isinstance(prop.get("banos"), (int, float)) else prop.get("banos"),
                                "imagen": prop.get("imagen"),
                                "descripcion": prop.get("descripcion") or "",
                            }
                            extra = None
                            if prop.get("score_total") is not None:
                                extra = f"Puntaje: {prop['score_total']:.3f}"
                            elif prop.get("similarity_score") is not None:
                                extra = f"Similitud: {prop['similarity_score']:.3f}"
                            render_card(pd.Series(row), extra_label=extra)
                            if isinstance(prop.get("url"), str) and prop.get("url"):
                                try:
                                    st.link_button("Ver aviso", prop["url"], use_container_width=True)
                                except Exception:
                                    st.write(f"[Ver aviso]({prop['url']})")
            except Exception as e:
                st.error(f"Error al procesar la solicitud: {e}")
                import traceback
                st.code(traceback.format_exc())

elif mood == "Mapa":
    try:
        import folium
        from streamlit_folium import st_folium
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError
        
        # Inicializar session_state para puntos seleccionados
        if "selected_points" not in st.session_state:
            st.session_state["selected_points"] = []
        
        # Función para geocodificar un lugar
        def geocode_location(query: str):
            """Busca coordenadas de un lugar usando Nominatim"""
            try:
                geolocator = Nominatim(user_agent="streamlit_property_app", timeout=10)
                location = geolocator.geocode(query, exactly_one=True)
                if location:
                    return {
                        "lat": location.latitude,
                        "lon": location.longitude,
                        "address": location.address
                    }
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                st.error(f"Error al buscar el lugar: {e}")
            except Exception as e:
                st.error(f"Error inesperado: {e}")
            return None
        
        st.write("Busque lugares por nombre o seleccione puntos en el mapa.")
        
        # Buscador de lugares
        col_search, col_add = st.columns([3, 1])
        with col_search:
            search_query = st.text_input("Buscar lugar (ej: UTN FRM, Plaza Independencia)", key="place_search")
        with col_add:
            st.write("")  # Espaciado
            search_button = st.button("Agregar", use_container_width=True)
        
        # Procesar búsqueda
        if search_button and search_query:
            with st.spinner(f"Buscando '{search_query}'..."):
                result = geocode_location(search_query)
                if result:
                    # Verificar si ya existe
                    exists = any(
                        abs(p["lat"] - result["lat"]) < 0.0001 and 
                        abs(p["lon"] - result["lon"]) < 0.0001
                        for p in st.session_state["selected_points"]
                    )
                    if not exists:
                        st.session_state["selected_points"].append(result)
                        st.success(f"✅ Agregado: {result['address']}")
                    else:
                        st.warning("Este lugar ya está en la lista.")
                else:
                    st.error("No se encontró el lugar. Intente con otro nombre.")
        
        # Sliders para alpha y sigma
        col_alpha, col_sigma = st.columns(2)
        with col_alpha:
            alpha = st.slider(
                "Alpha (importancia de características de la casa)",
                min_value=0.1,
                max_value=0.9,
                value=0.8,
                step=0.05,
                help="Más alto = más importancia a las características de la casa"
            )
        with col_sigma:
            sigma = st.slider(
                "Sigma (distancia de penalización en km)",
                min_value=1.0,
                max_value=20.0,
                value=4.0,
                step=0.5,
                help="Después de cuántos km promedio se penaliza la distancia"
            )
        
        # Crear mapa
        center = (-32.889, -68.845)
        if st.session_state["selected_points"]:
            # Centrar en el primer punto seleccionado
            center = (st.session_state["selected_points"][0]["lat"], 
                     st.session_state["selected_points"][0]["lon"])
        
        m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")
        folium.LatLngPopup().add_to(m)
        
        # Agregar marcadores para puntos seleccionados
        for idx, point in enumerate(st.session_state["selected_points"]):
            folium.Marker(
                location=[point["lat"], point["lon"]],
                popup=f"Punto {idx + 1}: {point.get('address', 'Sin dirección')}",
                tooltip=f"Click para eliminar",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
        
        # Mostrar mapa
        map_data = st_folium(m, height=420, use_container_width=True, key="main_map")
        
        # Manejar clicks en el mapa
        # Usar un key único para evitar procesar el mismo click múltiples veces
        if "last_processed_click" not in st.session_state:
            st.session_state["last_processed_click"] = None
        
        if map_data and map_data.get("last_clicked") and "lat" in map_data["last_clicked"] and "lng" in map_data["last_clicked"]:
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]
            
            # Verificar si este click ya fue procesado
            current_click = (clicked_lat, clicked_lon)
            if st.session_state["last_processed_click"] != current_click:
                st.session_state["last_processed_click"] = current_click
                
                # Verificar si se hizo click en un marcador existente (para eliminarlo)
                # Si el click está muy cerca de un punto existente, lo eliminamos
                clicked_on_marker = False
                for idx, point in enumerate(st.session_state["selected_points"]):
                    # Usar una distancia más pequeña para detectar clicks en marcadores
                    if abs(point["lat"] - clicked_lat) < 0.0005 and abs(point["lon"] - clicked_lon) < 0.0005:
                        # Eliminar punto
                        st.session_state["selected_points"].pop(idx)
                        st.success(f"✅ Punto eliminado: {point.get('address', 'Sin dirección')}")
                        st.rerun()
                        clicked_on_marker = True
                        break
                
                # Si no se hizo click en un marcador, agregar nuevo punto
                if not clicked_on_marker:
                    # Geocodificar inversa para obtener dirección
                    try:
                        geolocator = Nominatim(user_agent="streamlit_property_app", timeout=10)
                        location = geolocator.reverse((clicked_lat, clicked_lon), exactly_one=True)
                        address = location.address if location else f"Lat: {clicked_lat:.6f}, Lon: {clicked_lon:.6f}"
                    except Exception:
                        address = f"Lat: {clicked_lat:.6f}, Lon: {clicked_lon:.6f}"
                    
                    new_point = {
                        "lat": clicked_lat,
                        "lon": clicked_lon,
                        "address": address
                    }
                    
                    # Verificar si ya existe
                    exists = any(
                        abs(p["lat"] - new_point["lat"]) < 0.0001 and 
                        abs(p["lon"] - new_point["lon"]) < 0.0001
                        for p in st.session_state["selected_points"]
                    )
                    
                    if not exists:
                        st.session_state["selected_points"].append(new_point)
                        st.success(f"✅ Punto agregado: {address}")
                        st.rerun()
        
        # Mostrar lista de puntos seleccionados
        if st.session_state["selected_points"]:
            st.subheader("Puntos seleccionados:")
            points_to_remove = []
            for idx, point in enumerate(st.session_state["selected_points"]):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{idx + 1}.** {point.get('address', 'Sin dirección')}")
                    st.caption(f"Lat: {point['lat']:.6f}, Lon: {point['lon']:.6f}")
                with col2:
                    if st.button("Eliminar", key=f"remove_{idx}", use_container_width=True):
                        points_to_remove.append(idx)
            
            # Eliminar puntos marcados para eliminar
            for idx in sorted(points_to_remove, reverse=True):
                removed_point = st.session_state["selected_points"].pop(idx)
                st.success(f"✅ Punto eliminado: {removed_point.get('address', 'Sin dirección')}")
            if points_to_remove:
                st.rerun()
        else:
            st.info("No hay puntos seleccionados. Busque un lugar o haga click en el mapa.")
        
        # Parámetros de búsqueda
        modelo = "custom_embedding"
        output_qty = st.number_input("Cantidad de resultados", min_value=1, max_value=20, value=10, step=1)
        texto_busqueda = st.text_input("Texto de búsqueda (opcional)", value="", placeholder="Ej: casa con jardín")
        
        # Botón de búsqueda
        if st.button("Buscar propiedades", use_container_width=True):
            if not st.session_state["selected_points"]:
                st.warning("⚠️ Debe seleccionar al menos un punto en el mapa o buscar un lugar.")
            else:
                # Preparar coordenadas como objetos GeoPoint
                coordenadas = [
                    GeoPoint(lat=p["lat"], lon=p["lon"])
                    for p in st.session_state["selected_points"]
                ]
                
                with st.spinner("Consultando recomendaciones..."):
                    try:
                        input_data = UserInput(
                            texto=texto_busqueda,
                            modelo=modelo,
                            output_qty=int(output_qty),
                            coordenadas=coordenadas,
                            alpha=float(alpha),
                            sigma=float(sigma)
                        )
                        data_out = process_user_input(input_data)
                        
                        if data_out.get("status") == "error":
                            st.error(f"Error: {data_out.get('message', 'Error desconocido')}")
                        else:
                            props = (data_out or {}).get("output", {}).get("properties", [])
                            if not props:
                                st.info("Sin resultados.")
                            else:
                                st.subheader("Resultados:")
                                for prop in props:
                                    row = {
                                        "ciudad": prop.get("ciudad") or "",
                                        "ubicacion": prop.get("ubicacion") or prop.get("direccion") or "",
                                        "alquiler": prop.get("alquiler") if prop.get("alquiler") is not None else "",
                                        "m2_total": int(prop.get("m2_total")) if isinstance(prop.get("m2_total"), (int, float)) else prop.get("m2_total"),
                                        "ambientes": int(prop.get("ambientes")) if isinstance(prop.get("ambientes"), (int, float)) else prop.get("ambientes"),
                                        "banos": int(prop.get("banos")) if isinstance(prop.get("banos"), (int, float)) else prop.get("banos"),
                                        "imagen": prop.get("imagen"),
                                        "descripcion": prop.get("descripcion") or "",
                                    }
                                    extra = None
                                    if prop.get("score_total") is not None:
                                        extra = f"Puntaje: {prop['score_total']:.3f}"
                                    elif prop.get("similarity_score") is not None:
                                        extra = f"Similitud: {prop['similarity_score']:.3f}"
                                    if prop.get("distance_km") is not None:
                                        dist_text = f"Distancia: {prop['distance_km']:.2f} km"
                                        extra = f"{extra} | {dist_text}" if extra else dist_text
                                    render_card(pd.Series(row), extra_label=extra)
                                    if isinstance(prop.get("url"), str) and prop.get("url"):
                                        try:
                                            st.link_button("Ver aviso", prop["url"], use_container_width=True)
                                        except Exception:
                                            st.write(f"[Ver aviso]({prop['url']})")
                    except Exception as e:
                        st.error(f"Error al procesar la solicitud: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    except ImportError as e:
        st.error(f"Faltan dependencias necesarias: {e}")
        st.info("Instale las dependencias: 'folium', 'streamlit-folium' y 'geopy'.")
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
