"""
Property Recommender System

This module provides a property recommendation system with two model options:
1. sentence-transformers: Uses pre-trained models from sentence-transformers library
2. custom-embedding: Uses custom embeddings trained specifically with the dataset provided

The model selection is controlled via the --model command line argument. (FOR TESTING)
"""

import os
import re
import json
import pickle
import argparse
from typing import List, Dict, Optional
from datetime import datetime
import nltk

import numpy as np
import pandas as pd
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import math
from src.distance_calculator import haversine_km, DEFAULT_CITY, DEFAULT_COUNTRY
from dataclasses import dataclass

_USE_ST = False
try:
    from sentence_transformers import SentenceTransformer
    _USE_ST = True
except Exception:
    _USE_ST = False

_USE_EMBEDDING_MODEL = True

@dataclass
class GeoPoint:
    lat: float
    lon: float

DEFAULT_DATASET = r"src/alquiler_dataset.csv"

BASE_FIELDS = {
    "url","imagen","alquiler","expensas","ubicacion","m2_total","m2_cubiertos",
    "ambientes","banos","cocheras","dormitorios","antiguedad","disposicion",
    "orientacion","luminosidad","descripcion","direccion","ciudad","distrito"
}

ES_ADDITIONAL_STOP_WORDS = ['eramos', 'estabamos', 'estais', 'estan', 'estara', 'estaran', 'estaras', 'estare', 'estareis', 'estaria', 'estariais', 'estariamos', 'estarian', 'estarias', 'esteis', 'esten', 'estes', 'estuvieramos', 'estuviesemos', 'fueramos', 'fuesemos', 'habeis', 'habia', 'habiais', 'habiamos', 'habian', 'habias', 'habra', 'habran', 'habras', 'habre', 'habreis', 'habria', 'habriais', 'habriamos', 'habrian', 'habrias', 'hayais', 'hubieramos', 'hubiesemos', 'mas', 'mia', 'mias', 'mio', 'mios', 'seais', 'sera', 'seran', 'seras', 'sere', 'sereis', 'seria', 'seriais', 'seriamos', 'serian', 'serias', 'si', 'tambien', 'tendra', 'tendran', 'tendras', 'tendre', 'tendreis', 'tendria', 'tendriais', 'tendriamos', 'tendrian', 'tendrias', 'teneis', 'tengais', 'tenia', 'teniais', 'teniamos', 'tenian', 'tenias', 'tuvieramos', 'tuviesemos']

try:
    stopword_es = nltk.corpus.stopwords.words('spanish') + ES_ADDITIONAL_STOP_WORDS
except:
    stopword_es = []

# Function to normalize Spanish text
def normalize_text_es(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # lower case
    text = text.lower()
    # strip accents
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    # remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _safe_str(x: object) -> str:
    return str(x).strip() if pd.notna(x) else ""

# Function to build property description
def build_description(row: pd.Series) -> str:
    parts = []
    loc = _safe_str(row.get("ubicacion"))
    city = _safe_str(row.get("ciudad"))
    dist = _safe_str(row.get("distrito"))
    m2 = _safe_str(row.get("m2_total"))
    m2_cubiertos = _safe_str(row.get("m2_cubiertos"))
    rooms = _safe_str(row.get("ambientes"))
    beds = _safe_str(row.get("dormitorios"))
    baths = _safe_str(row.get("banos"))
    park = _safe_str(row.get("cocheras"))
    price = _safe_str(row.get("alquiler"))

    parts.append(f"Propiedad en {loc} ({city}/{dist}).")
    if m2:
        parts.append(f"Área total {m2} m².")
    if m2_cubiertos:
        parts.append(f"Área cubierta {m2_cubiertos} m².")
    if rooms:
        parts.append(f"Ambientes {rooms}.")
    if beds:
        parts.append(f"Dormitorios {beds}.")
    if baths:
        parts.append(f"Baños {baths}.")
    if park:
        parts.append(f"Cocheras {park}.")
    if price:
        parts.append(f"Alquiler {price}.")

    base_desc = _safe_str(row.get("descripcion"))
    if base_desc:
        parts.append(base_desc)

    return " ".join(parts)


"""-------- Sentence-Transformer Functions --------"""

# Function to generate embeddings for a list of texts
def embed_texts(texts: List[str]):
    """
    Generate embeddings for a list of texts using Sentence Transformers if available,
    otherwise use TF-IDF.
    
    Args:
        texts (List[str]): List of texts to vectorize
        
    Returns:
        np.ndarray: Matrix of embeddings (n_samples, n_features)
    """
    if _USE_ST:
        try:
            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            vecs = model.encode(texts, show_progress_bar=False)
            return np.array(vecs, dtype=np.float32)
        except Exception:
            pass

    # Fallback TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words=stopword_es, preprocessor=normalize_text_es)
    mat = tfidf.fit_transform(texts)
    return mat.toarray().astype(np.float32)


# Function to embed a single query using the same vectorizer
def embed_query(text: str, vectorizer: TfidfVectorizer):
    if _USE_ST:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return np.array(model.encode([text], show_progress_bar=False), dtype=np.float32)
    # Use vectorizer to transform query (sparse 1 x vocab)
    query_tfidf = vectorizer.transform([text])  # sparse 1 x vocab
    return query_tfidf


# Function to generate the path to save/load the model based on the current date
def get_model_path(model_type="custom_embedding"):
    """
    Generate the path to save/load the model based on the current date.
    
    Args:
        model_type (str): Model type ('custom_embedding' or 'sentence_transformer')
        
    Returns:
        str: Path to the model file
    """
    today = datetime.now().strftime("%Y%m%d")
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    return os.path.join(models_dir, f"{model_type}_{today}.pkl")


# Function to check if a trained model exists for the current date
def check_model_exists(model_type="custom_embedding"):
    """
    Check if a trained model exists for the current date.
    
    Args:
        model_type (str): Model type ('custom_embedding' or 'sentence_transformer')
        
    Returns:
        bool: True if model exists, False otherwise
    """
    model_path = get_model_path(model_type)
    return os.path.exists(model_path)


# Function to load a previously trained model
def load_model(model_type="custom_embedding"):
    """
    Load a previously trained model.
    
    Args:
        model_type (str): Model type ('custom_embedding' or 'sentence_transformer') 
        
    Returns:
        dict: Loaded model or None if it doesn't exist
    """
    model_path = get_model_path(model_type)
    if not os.path.exists(model_path):
        return None
        
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None


# Function to save a trained model
def save_model(model, model_type="custom_embedding"):
    """
    Save a trained model.
    
    Args:
        model: Model to save
        model_type (str): Model type ('custom_embedding' or 'sentence_transformer')
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    model_path = get_model_path(model_type)
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        return False

# train_custom_embedding
def train_custom_embedding(df: pd.DataFrame):
    """
    Train a custom embedding model with the provided dataset.
    
    Args:
        df (pandas.DataFrame): DataFrame with property data.
        
    Returns:
        dict: Model with embeddings and metadata.
    """
    # Create descriptions for all properties
    descriptions = [build_description(row) for _, row in df.iterrows()]
    
    # Train TfidfVectorizer on the complete corpus and obtain sparse embeddings
    vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1,2))
    corpus_tfidf = vectorizer.fit_transform(descriptions)  # sparse matrix
    
    # Create a model with vectorizer and sparse embeddings
    custom_model = {
        'embeddings': corpus_tfidf,           # sparse CSR matrix
        'vectorizer': vectorizer,             # TfidfVectorizer fitted
        'descriptions': descriptions,
        'property_ids': df.index.tolist(),
        'properties': df
    }
    
    return custom_model


def recommend_properties(user_input: str, model: str = "custom_embedding", output_qty: int = 5, data_path: str = DEFAULT_DATASET, alpha: float = 0.8, sigma: float = 4.0, city: str = DEFAULT_CITY, country: str = DEFAULT_COUNTRY, ref_points: Optional[List[Dict[str, float]]] = None) -> Dict:
    """
    Recommend properties using embeddings and/or Haversine distance based on input.

    - If there is text and coordinates: score_total = α * ((x + 1) / 2) + (1 - α) * exp(-(d^2) / (2 * σ^2))
    - If only text: score_total = ((x + 1) / 2)
    - If only coordinates: score_total = exp(-(d^2) / (2 * σ^2))

    Args:
        user_input (str): Text query (can be empty or None).
        model (str): 'custom_embedding' or 'sentence_transformer'.
        output_qty (int): Number of results.
        data_path (str): Path to the CSV dataset.
        alpha (float): Weight for textual similarity [0.6..0.9].
        sigma (float): Scale for distance decay (km).
        city (str): Default city for geocoding property addresses.
        country (str): Default country for geocoding property addresses.
        ref_points (List[Dict]): List of reference points [{'lat': float, 'lon': float}, ...].

    Returns:
        Dict: Results in JSON format.
    """
    import time
    t0 = time.perf_counter()
    times = {}
    
    # Verify dataset only if no model exists
    if not check_model_exists(model):
        if not os.path.exists(data_path):
            return {"error": f"Dataset not found: {data_path}"}

    try:
        df = None
        # Read CSV only if no model exists (avoid unnecessary reads)
        if not check_model_exists(model):
            t_csv = time.perf_counter()
            df = pd.read_csv(data_path)
            times['csv_read'] = time.perf_counter() - t_csv

        # Load/train model if it doesn't exist
        embedding_model = None
        if check_model_exists(model):
            t_load = time.perf_counter()
            embedding_model = load_model(model)
            times['model_load'] = time.perf_counter() - t_load
        # Validate model keys; if missing (e.g., 'vectorizer'), retrain
        needs_retrain = (
            embedding_model is None
            or (
                model == "custom_embedding"
                and not all(k in embedding_model for k in ["properties", "property_ids", "embeddings", "vectorizer"])  # ensure vectorizer
            )
        )
        if needs_retrain:
            t_train = time.perf_counter()
            embedding_model = train_custom_embedding(df if df is not None else pd.read_csv(data_path))
            save_model(embedding_model, model)
            times['model_train'] = time.perf_counter() - t_train

        # Extra: load df_coords for lat/lon even if model exists    
        df_coords = None
        try:
            df_tmp = pd.read_csv(data_path)
            if {'lat','lon'}.issubset(set(df_tmp.columns)):
                df_coords = df_tmp[['lat','lon']]
        except Exception:
            df_coords = None

        descriptions = embedding_model['descriptions']
        has_text = bool(user_input and str(user_input).strip())

        # Embedding only if there is a user's text
        sims = None
        if has_text:
            t_embed = time.perf_counter()
            if model == "sentence_transformer" and _USE_ST:
                query_emb = embed_texts([user_input])
                corpus_emb = embedding_model['embeddings']
            else:
                query_emb = embed_query(user_input, embedding_model['vectorizer'])
                corpus_emb = embedding_model['embeddings']
            times['embed_query'] = time.perf_counter() - t_embed

            t_sim = time.perf_counter()
            sims_mat = sklearn_cosine_similarity(corpus_emb, query_emb)
            sims = np.asarray(sims_mat).ravel()
            times['cosine_similarity'] = time.perf_counter() - t_sim

        # Normalize reference points (coordinates) if provided
        ref_points_list: List[GeoPoint] = []
        if ref_points:
            for p in ref_points:
                try:
                    # Handle both dict and object formats
                    if isinstance(p, dict):
                        lat = float(p.get("lat"))
                        lon = float(p.get("lon"))
                    else:
                        lat = float(p.lat)
                        lon = float(p.lon)
                    if pd.notna(lat) and pd.notna(lon):
                        ref_points_list.append(GeoPoint(lat=lat, lon=lon))
                except Exception:
                    continue

        # TOP-K properties
        if has_text and sims is not None:
            k = max(output_qty * 10, 50) 
            k = min(len(sims), k)
            candidate_idxs = np.argsort(-sims)[:k]
        else:
            # If no text, evaluate distance for all properties only
            candidate_idxs = np.arange(len(descriptions))

        results_buffer = []
        for i in candidate_idxs:
            # Get property details from dataset
            property_row = embedding_model['properties'].iloc[i].copy()

            # Use lat/lon from dataset for distance calculation
            prop_point: Optional[GeoPoint] = None
            if df_coords is not None:
                try:
                    lat_val = pd.to_numeric(df_coords.iloc[i].get('lat'), errors='coerce')
                    lon_val = pd.to_numeric(df_coords.iloc[i].get('lon'), errors='coerce')
                    if pd.notna(lat_val) and pd.notna(lon_val):
                        prop_point = GeoPoint(lat=lat_val, lon=lon_val)
                except Exception:
                    prop_point = None

            # If no lat/lon in dataset, cannot calculate distance
            d_km = None
            if ref_points_list and prop_point is not None:
                dists = []
                for rp in ref_points_list:
                    try:
                        dists.append(haversine_km(rp, prop_point))
                    except Exception:
                        continue
                if dists:
                    d_km = float(sum(dists) / len(dists))

            # Score based on similarity (only if text)
            x = None
            x_adj = None
            if has_text and sims is not None:
                x = float(sims[i])
                x_adj = (x + 1.0) / 2.0  # [0..1]

            # Puntaje total según disponibilidad de señales
            if has_text and x_adj is not None and d_km is not None and ref_points_list:
                dist_term = math.exp(- (d_km ** 2) / (2.0 * (sigma ** 2)))
                score_total = alpha * x_adj + (1.0 - alpha) * dist_term
            elif has_text and x_adj is not None:
                score_total = x_adj
            elif (not has_text) and d_km is not None and ref_points_list:
                dist_term = math.exp(- (d_km ** 2) / (2.0 * (sigma ** 2)))
                score_total = dist_term
            else:
                # Sin señales suficientes: omitir
                continue

            # Serializar propiedad
            prop_dict = property_row.to_dict()
            for kkey, vval in list(prop_dict.items()):
                if pd.isna(vval):
                    prop_dict[kkey] = None
                elif isinstance(vval, np.integer):
                    prop_dict[kkey] = int(vval)
                elif isinstance(vval, np.floating):
                    prop_dict[kkey] = float(vval)
            # Asegurar lat/lon en la respuesta desde dataset/prop_point
            if ('lat' not in prop_dict or prop_dict.get('lat') is None) or ('lon' not in prop_dict or prop_dict.get('lon') is None):
                if prop_point is not None:
                    prop_dict['lat'] = float(prop_point.lat)
                    prop_dict['lon'] = float(prop_point.lon)
                elif df_coords is not None:
                    try:
                        lat_val2 = pd.to_numeric(df_coords.iloc[i].get('lat'), errors='coerce')
                        lon_val2 = pd.to_numeric(df_coords.iloc[i].get('lon'), errors='coerce')
                        if pd.notna(lat_val2) and pd.notna(lon_val2):
                            prop_dict['lat'] = float(lat_val2)
                            prop_dict['lon'] = float(lon_val2)
                    except Exception:
                        pass
            prop_dict['similarity_score'] = x if x is not None else None
            prop_dict['similarity_score_adj'] = x_adj if x_adj is not None else None
            prop_dict['distance_km'] = float(d_km) if d_km is not None else None
            prop_dict['score_total'] = float(score_total)

            results_buffer.append((score_total, prop_dict))

        # Ordenar por score_total y tomar top N
        results_buffer.sort(key=lambda t: -t[0])
        results = [t[1] for t in results_buffer[:max(1, int(output_qty))]]

        times['total'] = time.perf_counter() - t0

        return {
            "query": user_input,
            "model": model,
            "alpha": alpha,
            "sigma": sigma,
            "results_count": len(results),
            "properties": results,
            "timing_ms": {k: round(v*1000,2) for k,v in times.items()}
        }

    except Exception as e:
        return {"error": f"Error al procesar la consulta: {str(e)}"}