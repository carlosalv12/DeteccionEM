from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from fastapi.middleware.cors import CORSMiddleware
import os
from huggingface_hub import InferenceApi

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    scores: dict

app = FastAPI(
    title="Sistema de detcción de enfermedades mentales",
    description="API para la detección de enfermedades mentales a partir de texto",
    version="1.0"
)


app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # o tu dominio
  allow_methods=["POST","GET"],
  allow_headers=["*"],
)

inference = InferenceApi(
   repo_id="carlosalv12/deteccionem-model",
   task = "text-classification",
   token=os.environ["HF_AUTH_TOKEN"]
)

def predict_labels(text: str):
    # Llamada al endpoint de inferencia de Hugging Face
    resp = inference({"inputs": text}, raw_response=True)
    out = resp.json()

    # Asumimos siempre lista de dicts [{label: ..., score: ...}, ...]
    if not isinstance(out, list) or len(out) == 0:
        raise ValueError(f"Respuesta inesperada de Inference API: {out}")

    # Tomamos el top-1
    top = out[0]
    label = top["label"]
    scores = {item["label"]: item["score"] for item in out}
    return label, scores

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="El campo 'text' no puede estar vacío")
    label, scores = predict_labels(request.text)
    return PredictionResponse(label=label, scores=scores)

#cd C:\Users\carlo\OneDrive\Escritorio\Master\TFM\
# .venv\Scripts\activate

#uvicorn --app-dir "C:\Users\carlo\OneDrive\Escritorio\Master\TFM" API_TFM:app --host 0.0.0.0 --port 8000 --reload
# uvicorn API_TFM:app --reload

#http://127.0.0.1:8000/docs