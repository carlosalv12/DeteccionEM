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
    # El formato puede variar según pipeline; aquí asumimos lista de dicts o dict con 'label' y 'scores'
    if isinstance(out, list):
        top = out[0]
        return top["label"], {top["label"]: top["score"]}
    # Si devuelve dict con 'label' y 'scores'
    return out["label"], out.get("scores", {})

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