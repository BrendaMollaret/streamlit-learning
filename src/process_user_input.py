from pydantic import BaseModel
from typing import Optional, List
import src.property_recommender as property_recommender

AVAILABLE_MODELS = ["custom_embedding", "sentence_transformer"]

class GeoPoint(BaseModel):
    lat: float
    lon: float

class UserInput(BaseModel):
    texto: str
    modelo: str
    output_qty: Optional[int] = 5
    coordenadas: Optional[List[GeoPoint]] = None
    alpha: Optional[float] = 0.8
    sigma: Optional[float] = 4.0

def process_user_input(input_data: UserInput):
    """
    Endpoint para procesar texto con un modelo específico.
    
    Parámetros:
    - texto: El texto a procesar
    - modelo: El modelo a utilizar para el procesamiento
    - output_qty: La cantidad de propiedades a recomendar (opcional, por defecto 5)
    - coordenadas: Las coordenadas geográficas para la recomendación (opcional)
        - lat: Latitud del punto geográfico
        - lon: Longitud del punto geográfico
    - alpha: Peso para la similitud textual [0.0-1.0] (opcional, por defecto 0.8)
    - sigma: Escala para la penalización por distancia en km (opcional, por defecto 4.0)
    
    Retorna:
    - Un objeto con la información del procesamiento
    """
    # extract text and model from input_data
    text = input_data.texto
    model = input_data.modelo
    output_qty = input_data.output_qty if input_data.output_qty else 5
    coordinates = input_data.coordenadas if input_data.coordenadas else None
    alpha = input_data.alpha if input_data.alpha is not None else 0.8
    sigma = input_data.sigma if input_data.sigma is not None else 4.0
    
    if model not in AVAILABLE_MODELS:
        return {"status": "error", "message": f"Modelo no disponible. Modelos disponibles: {AVAILABLE_MODELS}"}
    
    # Convert GeoPoint objects to dictionaries for recommend_properties
    ref_points = None
    if coordinates:
        ref_points = [{"lat": gp.lat, "lon": gp.lon} for gp in coordinates]
        
    recommend_properties = property_recommender.recommend_properties(
        text, 
        model, 
        output_qty, 
        ref_points=ref_points,
        alpha=alpha,
        sigma=sigma
    )
    if recommend_properties is None:
        return {"status": "error", "message": "Error al procesar la solicitud"}
    
    # Check if recommend_properties returned an error
    if isinstance(recommend_properties, dict) and "error" in recommend_properties:
        return {"status": "error", "message": recommend_properties.get("error", "Error desconocido")}
    
    return {
        "status": "success",
        "message": "Solicitud recibida",
        "input": {
            "texto": input_data.texto,
            "modelo": input_data.modelo,
            "output_qty": output_qty
        },
        "output": recommend_properties
    }