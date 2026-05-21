from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

import joblib
import pandas as pd

# Ruta del modelo
ruta_modelo = Path("models/best_random_forest.pkl")

# Cargar modelo
modelo = joblib.load(ruta_modelo)

# Inicializar API
app = FastAPI(
    title="API Predicción Deserción Escolar"
)

# Esquema de entrada
class DesercionInput(BaseModel):

    tasa_matriculacion_5_16: float
    cobertura_neta: float
    cobertura_neta_transicion: float
    cobertura_neta_primaria: float
    cobertura_neta_secundaria: float
    cobertura_neta_media: float
    cobertura_bruta: float
    cobertura_bruta_transicion: float
    cobertura_bruta_primaria: float
    cobertura_bruta_secundaria: float
    cobertura_bruta_media: float
    tamano_promedio_de_grupo: float
    aprobacion: float
    aprobacion_transicion: float
    aprobacion_primaria: float
    aprobacion_secundaria: float
    aprobacion_media: float
    reprobacion: float
    reprobacion_transicion: float
    reprobacion_primaria: float
    reprobacion_secundaria: float
    reprobacion_media: float
    repitencia: float
    repitencia_transicion: float
    repitencia_primaria: float
    repitencia_secundaria: float
    repitencia_media: float

# Endpoint principal
@app.post("/predict")
def predict(data: DesercionInput):

    try:

        input_data = pd.DataFrame([{
            "TASA_MATRICULACIÓN_5_16": data.tasa_matriculacion_5_16,
            "COBERTURA_NETA": data.cobertura_neta,
            "COBERTURA_NETA_TRANSICIÓN": data.cobertura_neta_transicion,
            "COBERTURA_NETA_PRIMARIA": data.cobertura_neta_primaria,
            "COBERTURA_NETA_SECUNDARIA": data.cobertura_neta_secundaria,
            "COBERTURA_NETA_MEDIA": data.cobertura_neta_media,
            "COBERTURA_BRUTA": data.cobertura_bruta,
            "COBERTURA_BRUTA_TRANSICIÓN": data.cobertura_bruta_transicion,
            "COBERTURA_BRUTA_PRIMARIA": data.cobertura_bruta_primaria,
            "COBERTURA_BRUTA_SECUNDARIA": data.cobertura_bruta_secundaria,
            "COBERTURA_BRUTA_MEDIA": data.cobertura_bruta_media,
            "TAMAÑO_PROMEDIO_DE_GRUPO": data.tamano_promedio_de_grupo,
            "APROBACIÓN": data.aprobacion,
            "APROBACIÓN_TRANSICIÓN": data.aprobacion_transicion,
            "APROBACIÓN_PRIMARIA": data.aprobacion_primaria,
            "APROBACIÓN_SECUNDARIA": data.aprobacion_secundaria,
            "APROBACIÓN_MEDIA": data.aprobacion_media,
            "REPROBACIÓN": data.reprobacion,
            "REPROBACIÓN_TRANSICIÓN": data.reprobacion_transicion,
            "REPROBACIÓN_PRIMARIA": data.reprobacion_primaria,
            "REPROBACIÓN_SECUNDARIA": data.reprobacion_secundaria,
            "REPROBACIÓN_MEDIA": data.reprobacion_media,
            "REPITENCIA": data.repitencia,
            "REPITENCIA_TRANSICIÓN": data.repitencia_transicion,
            "REPITENCIA_PRIMARIA": data.repitencia_primaria,
            "REPITENCIA_SECUNDARIA": data.repitencia_secundaria,
            "REPITENCIA_MEDIA": data.repitencia_media
        }])

        prediction = modelo.predict(input_data)

        return {
            "prediccion_desercion": int(prediction[0])
        }

    except Exception as e:

        return {
            "error": str(e)
        }
