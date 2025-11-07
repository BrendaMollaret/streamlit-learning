import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO
import os
from math import radians, sin, cos, asin, sqrt, pi
import base64
from src.process_user_input import process_user_input, UserInput, GeoPoint

# Configurar el layout de la página para usar ancho completo
st.set_page_config(layout="wide", page_title="Buscador de Propiedades Inteligente")

# CSS personalizado para aumentar el tamaño de los textos y el ancho del contenido
st.markdown("""
    <style>
    /* Aumentar ancho máximo del contenido principal - dinámico y centrado */
    .main .block-container {
        max-width: 80vw !important;
        padding-left: 2.5% !important;
        padding-right: 2.5% !important;
    }
    
    /* Asegurar que los elementos ocupen el espacio disponible */
    .element-container {
        max-width: 100% !important;
    }
    
    /* Aumentar tamaño de texto general */
    .stMarkdown, .stMarkdown p {
        font-size: 0.95rem !important;
    }
    
    /* Títulos más grandes */
    h1 {
        font-size: 2rem !important;
    }
    
    h2 {
        font-size: 1.6rem !important;
    }
    
    h3 {
        font-size: 1.4rem !important;
    }
    
    /* Texto en las tarjetas */
    .element-container .stMarkdown p {
        font-size: 1rem !important;
    }
    
    /* Captions más grandes */
    .stCaption {
        font-size: 0.9rem !important;
    }
    
    /* Texto en los inputs */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        font-size: 1rem !important;
    }
    
    /* Texto en los botones */
    .stButton > button {
        font-size: 1rem !important;
    }
    
    /* Reducir espacios verticales entre inputs, sliders y botones */
    .block-container div[data-testid="stVerticalBlock"] > div {
    margin-bottom: 0.4rem !important;
    }

    /* Reducir padding interno de contenedores con borde */
    div[data-testid="stContainer"] {
    padding-top: 0.3rem !important;
    padding-bottom: 0.3rem !important;
    }

    /* Ajustar separación en sliders */
    [data-baseweb="slider"] {
    margin-top: 0.3rem !important;
    margin-bottom: 0.3rem !important;
    }

    /* Reducir margen de subtítulos y textos */
    h2, h3, label, p, .stMarkdown {
    margin-top: 0.3rem !important;
    margin-bottom: 0.3rem !important;
    }

    /* Compactar los inputs numéricos y selectboxes */
    .stNumberInput, .stSelectbox, .stTextInput {
    margin-bottom: 0.5rem !important;
    }

    /* --- Ajustar separación entre controles específicos --- */

    /* Reducir espacio entre selectbox y text_input */
    div[data-testid="stSelectbox"] + div[data-testid="stTextInput"] {
    margin-top: -0.4rem !important;
    }

    /* Reducir espacio entre los sliders Alpha y Sigma */
    div[data-testid="stSlider"]:has(label:contains("Alpha")) {
    margin-bottom: 0.1rem !important;
    }
    div[data-testid="stSlider"]:has(label:contains("Sigma")) {
    margin-top: -0.1rem !important;
    }

    /* También ajustar los divisores cercanos */
    hr {
    margin-top: 0.3rem !important;
    margin-bottom: 0.3rem !important;
    }

    /* Responsive: en móvil las columnas se apilan */
    @media (max-width: 768px) {
        .element-container [data-testid="column"] {
            width: 100% !important;
            flex: 0 0 100% !important;
        }
        .main .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        /* Asegurar que las columnas se apilen en móvil */
        [data-testid="column"] {
            width: 100% !important;
            flex: 0 0 100% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header centrado
st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 2.5rem; font-weight: bold; background: linear-gradient(to right, #1f77b4, #ff7f0e); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.5rem;">
            Buscador de Propiedades Inteligente
        </h1>
        <p style="color: #666; font-size: 1rem; margin-top: 0.5rem;">
            Encuentra tu propiedad ideal usando búsqueda inteligente por mapa
        </p>
    </div>
""", unsafe_allow_html=True)

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

def offset_lat_lon(lat: float, lon: float, i: int, meters: float = 6.0) -> tuple[float, float]:
    """Return a small offset lat/lon around the original point using i as an index.

    i=0 returns the original point (no offset). For i>0 returns points arranged
    on a small circle with radius proportional to meters * i.
    """
    if i <= 0:
        return lat, lon
    # angle spread using the index to avoid stacking
    angle = (i) * (2 * pi / 8)
    dx = meters * cos(angle)
    dy = meters * sin(angle)
    # meters to degrees conversion
    dlat = dy / 111111.0
    # adjust lon conversion by latitude
    lat_rad = radians(lat)
    meters_per_deg_lon = 111111.0 * cos(lat_rad) if cos(lat_rad) != 0 else 111111.0
    dlon = dx / meters_per_deg_lon
    return lat + dlat, lon + dlon

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


def render_card(row: pd.Series, extra_label: str | None = None, show_number: bool = True, number: int | None = None) -> None:
    with st.container(border=True):
        img = prepare_card_image(row["imagen"]) if isinstance(row["imagen"], str) else None
        render_full_width_image(img, fallback_url=row.get("imagen") if isinstance(row.get("imagen"), str) else None)
        title_prefix = f"{number} - " if show_number and number is not None else ""
        st.markdown(f"<h2 style='font-size: 1.4rem; margin-bottom: 0.5rem;'>{title_prefix}{row['ciudad']} – {row['ubicacion']}</h2>", unsafe_allow_html=True)
        if extra_label:
            st.markdown(f"<p style='font-size: 0.95rem; color: #666; margin-top: 0;'>{extra_label}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem; margin: 0.5rem 0;'>💰 <strong>Alquiler:</strong> ${row['alquiler']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1rem; margin: 0.5rem 0;'>📏 {row['m2_total']} m² | 🛏 {row['ambientes']} amb | 🚿 {row['banos']} baños</p>", unsafe_allow_html=True)
        # Truncar descripción a 100 caracteres para que todas las cards tengan el mismo tamaño
        descripcion = str(row['descripcion']) if row.get('descripcion') else ''
        max_chars = 100
        if len(descripcion) > max_chars:
            descripcion = descripcion[:max_chars].rsplit(' ', 1)[0] + '...'
        st.markdown(f"<p style='font-size: 0.95rem; margin: 0.5rem 0;'>📝 {descripcion}</p>", unsafe_allow_html=True)

# Modo Mapa siempre activo
try:
    import folium
    from streamlit_folium import st_folium
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    
    # Inicializar session_state para puntos seleccionados y propiedades encontradas
    if "selected_points" not in st.session_state:
        st.session_state["selected_points"] = []
    if "found_properties" not in st.session_state:
        st.session_state["found_properties"] = []
    if "modo_texto" not in st.session_state:
        st.session_state["modo_texto"] = "Texto libre"
    
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
    
    # Estructura de dos columnas: izquierda (controles) y derecha (mapa)
    # Usar proporciones similares al React: izquierda ~400px, derecha el resto
    col_left, col_right = st.columns([1, 2.5], gap="large")
    
    with col_left:
        # Card con controles de búsqueda
        with st.container(border=True):
            # Modo de texto de búsqueda
            modo_texto = st.selectbox(
                "Modo de texto de búsqueda",
                ["Texto libre", "Parámetros"],
                index=0 if st.session_state["modo_texto"] == "Texto libre" else 1,
                key="modo_texto_selector"
            )
            st.session_state["modo_texto"] = modo_texto
            
            # Mostrar controles según el modo seleccionado
            texto_busqueda = ""
            
            if modo_texto == "Texto libre":
                # Búsqueda por texto libre
                texto_busqueda = st.text_input(
                    "Texto de búsqueda",
                    value="",
                    placeholder="Ej: departamento con jardín y pileta",
                    help="Busque propiedades por características",
                    key="texto_busqueda_mapa"
                )
            else:  # Parámetros
                # Controles de parámetros
                col1, col2 = st.columns(2)
                with col1:
                    min_m2 = st.number_input("Mín. m²", min_value=10, value=10, step=10, key="min_m2_mapa")
                    min_amb = st.number_input("Mín. ambientes", min_value=1, value=1, step=1, key="min_amb_mapa")
                    min_banos = st.number_input("Mín. baños", min_value=1, value=1, step=1, key="min_banos_mapa")
                with col2:
                    max_m2 = st.number_input("Máx. m²", min_value=10, value=10, step=10, key="max_m2_mapa")
                    min_cocheras = st.number_input("Mín. cocheras", min_value=1, value=1, step=1, key="min_cocheras_mapa")
                    min_alq = st.number_input("Alquiler mín.", min_value=0, value=0, step=1000, key="min_alq_mapa")
                    max_alq = st.number_input("Alquiler máx.", min_value=0, value=0, step=1000, key="max_alq_mapa")
                
                # Construir texto de consulta a partir de los parámetros
                partes = []
                if min_m2 > 0 and max_m2 > 0:
                    if max_m2 >= min_m2:
                        partes.append(f"entre {int(min_m2)} y {int(max_m2)} m²")
                    else:
                        partes.append(f"al menos {int(min_m2)} m²")
                elif min_m2 > 0:
                    partes.append(f"al menos {int(min_m2)} m²")
                elif max_m2 > 0:
                    partes.append(f"hasta {int(max_m2)} m²")
                if min_amb > 0:
                    partes.append(f"con al menos {int(min_amb)} ambientes")
                if min_banos > 0:
                    partes.append(f"con al menos {int(min_banos)} baños")
                if min_cocheras > 0:
                    partes.append(f"con al menos {int(min_cocheras)} cocheras")
                if min_alq > 0 and max_alq > 0:
                    if max_alq >= min_alq:
                        partes.append(f"alquiler entre {int(min_alq)} y {int(max_alq)}")
                    else:
                        partes.append(f"alquiler desde {int(min_alq)}")
                elif min_alq > 0:
                    partes.append(f"alquiler desde {int(min_alq)}")
                elif max_alq > 0:
                    partes.append(f"alquiler hasta {int(max_alq)}")
                
                texto_busqueda = " ".join(partes).strip()
                if not texto_busqueda:
                    texto_busqueda = ""  # Dejar vacío si no hay parámetros
            
            # Separador
            st.divider()
            
            # Sliders para alpha y sigma
            alpha = st.slider(
                "¿Preferís que esté cerca de tus puntos o que cumpla más con lo que buscás?",
                min_value=0.1,
                max_value=0.9,
                value=0.8,
                step=0.05,
                help="Subí el valor si querés priorizar las características de la casa (como jardín, pileta, tamaño, etc.) por sobre la ubicación.",
                key="alpha_mapa"
            )

            sigma = st.slider(
                "¿Hasta qué distancia te gustaría que busquemos alrededor de tus puntos?",
                min_value=1.0,
                max_value=20.0,
                value=4.0,
                step=0.5,
                help="Un valor más alto amplía la zona de búsqueda.",
                key="sigma_mapa"
            )

            
            # Separador
            st.divider()
            
            # Cantidad de resultados
            output_qty = st.number_input(
                "Cantidad de resultados",
                min_value=1,
                max_value=20,
                value=10,
                step=1,
                key="output_qty_mapa"
            )
            
            # Botón de búsqueda
            modelo = "custom_embedding"
            if st.button("Buscar propiedades", use_container_width=True, key="buscar_propiedades_mapa"):
                # Puede buscar solo por texto, solo por puntos, o ambos
                if not texto_busqueda.strip() and not st.session_state["selected_points"]:
                    # Mostrar advertencia como toast para no romper el layout
                    try:
                        st.toast("Debe proporcionar texto de búsqueda o seleccionar puntos en el mapa (o ambos).", icon="⚠️")
                    except Exception:
                        # Fallback si st.toast no está disponible en la versión de streamlit
                        st.warning("Debe proporcionar texto de búsqueda o seleccionar puntos en el mapa (o ambos).")
                else:
                    # Preparar coordenadas como objetos GeoPoint (si hay puntos)
                    coordenadas = None
                    if st.session_state["selected_points"]:
                        coordenadas = [
                            GeoPoint(lat=p["lat"], lon=p["lon"])
                            for p in st.session_state["selected_points"]
                        ]
                    
                    with st.spinner("Consultando recomendaciones..."):
                        try:
                            input_data = UserInput(
                                texto=texto_busqueda if texto_busqueda.strip() else "",
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
                                # Get properties from the output
                                output = data_out.get("output", {})
                                if isinstance(output, dict) and "error" in output:
                                    st.error(f"Error: {output.get('error', 'Error desconocido')}")
                                else:
                                    props = output.get("properties", []) if isinstance(output, dict) else []
                                    if not props:
                                        st.info("Sin resultados.")
                                        st.session_state["found_properties"] = []
                                    else:
                                        # Guardar propiedades en session_state para mostrarlas en el mapa
                                        st.session_state["found_properties"] = props
                                        st.success(f"✅ Se encontraron {len(props)} propiedades. Ver el mapa y el listado abajo.")
                                        st.rerun()
                        except Exception as e:
                            st.error(f"Error al procesar la solicitud: {e}")
                            import traceback
                            st.code(traceback.format_exc())
            
            # Puntos seleccionados (debajo del card)
            if st.session_state["selected_points"]:
                # Usar markdown con estilo más pequeño en lugar de st.subheader
                st.markdown(
                    f"<h3 id='puntos-seleccionados' style='font-size:1rem; margin-bottom:0.25rem; margin-top:0.5rem;'>Puntos seleccionados ({len(st.session_state['selected_points'])})</h3>",
                    unsafe_allow_html=True,
                )
                points_to_remove = []
                for idx, point in enumerate(st.session_state["selected_points"]):
                    address = point.get('address', 'Sin dirección')
                    # Truncar dirección: tomar solo las primeras partes hasta "Boulogne Sur Mer," o las primeras 3 partes
                    address_parts = address.split(', ')
                    if len(address_parts) > 3:
                        # Tomar las primeras 3 partes (hasta "Boulogne Sur Mer,")
                        short_address = ', '.join(address_parts[:3]) + ','
                    else:
                        short_address = address
                    # Texto más pequeño para la lista de puntos seleccionados
                    st.markdown(
                        f"<p style='font-size: 0.75rem; margin: 0.05rem 0; color: #333;'>{short_address}</p>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<p style='font-size: 0.65rem; color: #666; margin-top: 0; margin-bottom: 0.4rem;'>"
                        f"Lat: {point['lat']:.6f}, Lon: {point['lon']:.6f}</p>",
                        unsafe_allow_html=True
                    )
                    if st.button("Eliminar", key=f"remove_{idx}", use_container_width=True):
                        points_to_remove.append(idx)
                
                # Eliminar puntos marcados para eliminar
                for idx in sorted(points_to_remove, reverse=True):
                    removed_point = st.session_state["selected_points"].pop(idx)
                    st.success(f"✅ Punto eliminado: {removed_point.get('address', 'Sin dirección')}")
                if points_to_remove:
                    st.rerun()
        
        with col_right:
            # Mapa en la columna derecha
            # Crear mapa y calcular centro y bounds
            all_locations = []
            
            # Agregar puntos seleccionados
            for point in st.session_state["selected_points"]:
                all_locations.append([point["lat"], point["lon"]])
            
            # Agregar propiedades encontradas
            for prop in st.session_state["found_properties"]:
                if prop.get("lat") is not None and prop.get("lon") is not None:
                    try:
                        lat = float(prop["lat"])
                        lon = float(prop["lon"])
                        if pd.notna(lat) and pd.notna(lon):
                            all_locations.append([lat, lon])
                    except (ValueError, TypeError):
                        continue
            
            # Calcular centro y zoom
            if all_locations:
                # Calcular bounds
                lats = [loc[0] for loc in all_locations]
                lons = [loc[1] for loc in all_locations]
                center = (sum(lats) / len(lats), sum(lons) / len(lons))
                
                # Calcular zoom apropiado basado en la dispersión
                lat_range = max(lats) - min(lats)
                lon_range = max(lons) - min(lons)
                max_range = max(lat_range, lon_range)
                
                if max_range > 0.1:
                    zoom_start = 10
                elif max_range > 0.05:
                    zoom_start = 11
                elif max_range > 0.02:
                    zoom_start = 12
                else:
                    zoom_start = 13
            else:
                center = (-32.889, -68.845)
                zoom_start = 11
            
            m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")
            folium.LatLngPopup().add_to(m)
            
            # Agregar control de búsqueda directamente en el mapa con estilo personalizado
            try:
                from folium.plugins import Geocoder
                
                # Agregar CSS personalizado para hacer el control de búsqueda más grande
                geocoder_html = """
                <style>
                .leaflet-control-geocoder {
                    font-size: 18px !important;
                }
                .leaflet-control-geocoder input {
                    font-size: 18px !important;
                    padding: 10px !important;
                    height: 45px !important;
                    min-width: 300px !important;
                }
                .leaflet-control-geocoder a {
                    font-size: 18px !important;
                    padding: 10px !important;
                }
                .leaflet-control-geocoder-form {
                    font-size: 18px !important;
                }
                </style>
                """
                m.get_root().html.add_child(folium.Element(geocoder_html))
                
                # Crear Geocoder - cuando se busca un lugar, agrega un marcador automáticamente
                geocoder = Geocoder(
                    collapsed=False,
                    position='topleft',
                    add_marker=True
                )
                geocoder.add_to(m)
                
                # Agregar JavaScript para capturar automáticamente cuando el Geocoder agrega un marcador
                geocoder_script = """
                <script>
                (function() {
                    setTimeout(function() {
                        if (typeof L !== 'undefined') {
                            var maps = document.querySelectorAll('.folium-map');
                            maps.forEach(function(mapElement) {
                                var map = mapElement._leaflet_id ? L.map(mapElement._leaflet_id) : null;
                                if (map) {
                                    map.on('geocoder_result', function(e) {
                                        if (e.result && e.result.center) {
                                            var lat = e.result.center.lat;
                                            var lng = e.result.center.lng;
                                            var name = e.result.name || '';
                                            
                                            var hiddenInput = document.getElementById('geocoder_result');
                                            if (!hiddenInput) {
                                                hiddenInput = document.createElement('input');
                                                hiddenInput.type = 'hidden';
                                                hiddenInput.id = 'geocoder_result';
                                                document.body.appendChild(hiddenInput);
                                            }
                                            hiddenInput.value = JSON.stringify({lat: lat, lng: lng, name: name});
                                            hiddenInput.dispatchEvent(new Event('change'));
                                        }
                                    });
                                }
                            });
                        }
                    }, 2000);
                })();
                </script>
                """
                m.get_root().html.add_child(folium.Element(geocoder_script))
            except ImportError:
                pass
            
            # Si hay múltiples ubicaciones, ajustar bounds
            if len(all_locations) > 1:
                m.fit_bounds(all_locations)
            
            # Preparar offsets para puntos seleccionados (evitar solapamiento)
            sel_points = st.session_state["selected_points"]
            sel_counts = {}
            for p in sel_points:
                key = (round(float(p.get("lat", 0)), 6), round(float(p.get("lon", 0)), 6))
                sel_counts[key] = sel_counts.get(key, 0) + 1

            sel_seen = {}
            # Agregar marcadores para puntos seleccionados (rojos)
            for idx, point in enumerate(sel_points):
                address = point.get('address', 'Sin dirección')
                popup_html_ref = f"""
                <div style="min-width: 220px; font-family: Arial, sans-serif; padding: 4px;">
                    <h3 style="font-size: 14px; font-weight: bold; margin: 4px 0; color: #d32f2f;">📍 Punto</h3>
                    <p style="font-size: 12px; margin: 6px 0; line-height: 1.3;">{address}</p>
                    <p style="font-size: 11px; margin: 4px 0; color: #666;">Click en el mapa para eliminar</p>
                </div>
                """
                # Aplicar pequeño offset si hay varios puntos en la misma coordenada
                try:
                    lat0 = float(point["lat"])
                    lon0 = float(point["lon"])
                except Exception:
                    lat0, lon0 = point.get("lat"), point.get("lon")
                key = (round(float(lat0), 6), round(float(lon0), 6))
                seen = sel_seen.get(key, 0)
                sel_seen[key] = seen + 1
                lat_off, lon_off = offset_lat_lon(lat0, lon0, seen)

                # Usar icon más pequeño y menos padding (sin número)
                folium.Marker(
                    location=[lat_off, lon_off],
                    popup=folium.Popup(popup_html_ref, max_width=260),
                    tooltip=folium.Tooltip("Punto de referencia", sticky=True),
                    icon=folium.DivIcon(
                        html=f'<div style="background-color: #d9534f; border-radius: 50%; width: 32px; height: 32px; display: block; border: 3px solid white;"></div>',
                        icon_size=(32, 32),
                        icon_anchor=(16, 16)
                    ),
                    zIndexOffset=0
                ).add_to(m)
            
            # Preparar offsets para propiedades encontradas (evitar solapamiento)
            props = st.session_state["found_properties"]
            prop_counts = {}
            for p in props:
                if p.get("lat") is None or p.get("lon") is None:
                    continue
                key = (round(float(p.get("lat", 0)), 6), round(float(p.get("lon", 0)), 6))
                prop_counts[key] = prop_counts.get(key, 0) + 1

            prop_seen = {}
            # Agregar marcadores para propiedades encontradas (azules con números)
            for idx, prop in enumerate(props, 1):
                if prop.get("lat") is not None and prop.get("lon") is not None:
                    try:
                        lat = float(prop["lat"])
                        lon = float(prop["lon"])
                        if pd.notna(lat) and pd.notna(lon):
                            # Crear popup con información de la propiedad (más legible)
                            ciudad = prop.get('ciudad', 'N/A')
                            ubicacion = prop.get('ubicacion', prop.get('direccion', 'N/A'))
                            alquiler = prop.get('alquiler', 'N/A')
                            m2 = prop.get('m2_total', 'N/A')
                            ambientes = prop.get('ambientes', 'N/A')
                            banos = prop.get('banos', 'N/A')
                            descripcion = prop.get('descripcion', 'Sin descripción')
                            distancia = f'{prop.get("distance_km", 0):.2f} km' if prop.get('distance_km') is not None else None
                            puntaje = f'{prop.get("score_total", 0):.3f}' if prop.get('score_total') is not None else None
                            
                            # Tooltip cuadrado, más ancho y alto para mostrar texto largo
                            tooltip_html = f"""
                                <div style="width: 300px; height: 120px; font-family: Arial, sans-serif; white-space: normal; word-wrap: break-word; overflow-wrap: break-word; display: flex; flex-direction: column; justify-content: center; align-items: flex-start;">
                                    <h4 style="font-size: 15px; font-weight: bold; margin: 6px 0; color: #1f77b4;">#{idx} - {ciudad}</h4>
                                    <div style="font-size: 14px; margin: 6px 0; line-height: 1.2;">Ubicación:<br/>{ubicacion}</div>
                                    <div style="font-size: 14px; margin: 6px 0; color: #2e7d32; font-weight: bold;">Alquiler:<br/>${alquiler}</div>
                                </div>
                            """
                            
                            popup_html = f"""
                            <div style="min-width: 280px; font-family: Arial, sans-serif; padding: 5px;">
                                <h3 style="font-size: 18px; font-weight: bold; margin: 5px 0; color: #1f77b4;">#{idx} - {ciudad}</h3>
                                <p style="font-size: 15px; margin: 8px 0; line-height: 1.4;"><b style="font-size: 15px;">Ubicación:</b><br>{ubicacion}</p>
                                <p style="font-size: 16px; margin: 8px 0; color: #2e7d32; font-weight: bold;"><b>Alquiler:</b> ${alquiler}</p>
                                <p style="font-size: 14px; margin: 8px 0; line-height: 1.5;"><b>m²:</b> {m2} | <b>Amb:</b> {ambientes} | <b>Baños:</b> {banos}</p>
                                <p style="font-size: 14px; margin: 8px 0; line-height: 1.4;"><b>Descripción:</b><br>{descripcion}</p>
                                {f'<p style="font-size: 14px; margin: 8px 0;"><b>Distancia:</b> {distancia}</p>' if distancia else ''}
                                {f'<p style="font-size: 14px; margin: 8px 0;"><b>Puntaje:</b> {puntaje}</p>' if puntaje else ''}
                            </div>
                            """
                            # Aplicar pequeño offset si hay varias propiedades en la misma coordenada
                            keyp = (round(float(lat), 6), round(float(lon), 6))
                            seenp = prop_seen.get(keyp, 0)
                            prop_seen[keyp] = seenp + 1
                            latp, lonp = offset_lat_lon(lat, lon, seenp, meters=8.0)

                            folium.Marker(
                                location=[latp, lonp],
                                popup=folium.Popup(popup_html, max_width=350),
                                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                                icon=folium.DivIcon(
                                    html=f'<div style="background-color: #3388ff; color: white; border-radius: 50%; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 16px; border: 3px solid white; box-shadow: 0 3px 6px rgba(0,0,0,0.4);">{idx}</div>',
                                    icon_size=(35, 35),
                                    icon_anchor=(17, 17)
                                ),
                                zIndexOffset=1000
                            ).add_to(m)
                    except (ValueError, TypeError):
                        continue
            
            # Mostrar mapa
            # Agregar leyenda: rojo = puntos de referencia, azul = propiedades encontradas
            legend_html = '''
                <div style="position: fixed; bottom: 80px; left: 10px; z-index:9999; background: white; padding: 8px; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15); font-size:13px;">
                    <div style="display:flex;align-items:center;margin-bottom:4px;"><div style='width:14px;height:14px;background:#d9534f;border-radius:50%;margin-right:8px;border:2px solid #fff;'></div>Puntos de referencia</div>
                    <div style="display:flex;align-items:center;"><div style='width:16px;height:16px;background:#3388ff;border-radius:50%;margin-right:8px;border:2px solid #fff;'></div>Propiedades encontradas</div>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))

            map_data = st_folium(m, height=800, use_container_width=True, key="main_map", returned_objects=["last_clicked", "last_object_clicked"])
        
        # Detectar marcadores agregados por el Geocoder
        # El Geocoder agrega marcadores automáticamente, pero necesitamos detectarlos
        # Usamos last_object_clicked para detectar cuando se hace click en un marcador del Geocoder
        if map_data and map_data.get("last_object_clicked"):
            obj_clicked = map_data.get("last_object_clicked")
            if obj_clicked and "lat" in obj_clicked and "lng" in obj_clicked:
                clicked_lat = obj_clicked["lat"]
                clicked_lon = obj_clicked["lng"]
                
                # Verificar si este punto ya está en la lista
                exists = any(
                    abs(p["lat"] - clicked_lat) < 0.0001 and 
                    abs(p["lon"] - clicked_lon) < 0.0001
                    for p in st.session_state["selected_points"]
                )
                
                if not exists:
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
                        # Limitar a 3 puntos
                    if len(st.session_state["selected_points"]) >= 3:
                        st.toast("⚠️ Solo puedes marcar hasta 3 puntos de referencia.", icon="⚠️")
                    else:
                        st.session_state["selected_points"].append(new_point)
                        st.rerun()
                  
        
        # Manejar clicks en el mapa
        # Usar un key único para evitar procesar el mismo click múltiples veces
        if "last_processed_click" not in st.session_state:
            st.session_state["last_processed_click"] = None
        
    # Evitar procesar last_clicked si last_object_clicked está presente (clicks en marcadores)
    if map_data and not map_data.get("last_object_clicked") and map_data.get("last_clicked") and "lat" in map_data["last_clicked"] and "lng" in map_data["last_clicked"]:
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
                        # Limitar a 3 puntos
                        if len(st.session_state["selected_points"]) >= 3:
                            st.toast("Solo puedes marcar hasta 3 puntos de referencia.", icon="⚠️")
                        else:
                            st.session_state["selected_points"].append(new_point)
                            st.rerun()
    
    # Mostrar propiedades encontradas debajo del mapa (en columnas de 3)
    if st.session_state["found_properties"]:
        st.subheader("Propiedades encontradas:")
        props = st.session_state["found_properties"]
        # Mostrar propiedades en columnas de 3
        for i in range(0, len(props), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(props):
                    prop = props[i + j]
                    number = i + j + 1  # numeración para correlacionar con marcadores del mapa
                    with col:
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
                        # Mostrar sin numeración (show_number=False)
                        # Mostrar numeración para correlacionar con el mapa
                        render_card(pd.Series(row), extra_label=extra, show_number=True, number=number)
                        if isinstance(prop.get("url"), str) and prop.get("url"):
                            try:
                                st.link_button("Ver aviso", prop["url"], use_container_width=True)
                            except Exception:
                                st.write(f"[Ver aviso]({prop['url']})")
    
    # Mensaje de ayuda
    if not st.session_state["selected_points"] and not st.session_state["found_properties"]:
        st.info("💡 **Tip:** Selecciona puntos en el mapa haciendo click o busca lugares usando el campo de texto")
except ImportError as e:
    st.error(f"Faltan dependencias necesarias: {e}")
    st.info("Instale las dependencias: 'folium', 'streamlit-folium' y 'geopy'.")
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())
