from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from fastapi.middleware.cors import CORSMiddleware
import os

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

HF_MODEL = "carlosalv12/deteccionem-model"

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(
               HF_MODEL,
               low_cpu_mem_usage=True
            )
model.eval()

def predict_labels(text: str):
    # Tokenización y creación de tensores
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logs = model(**inputs).logits
    # Probabilidades softmax
    probs = torch.softmax(logs, dim=-1).squeeze().tolist()
    # Mapear ids a etiquetas
    id2label = model.config.id2label
    scores = {id2label[i]: probs[i] for i in range(len(probs))}
    # Seleccionar etiqueta con mayor probabilidad
    label = max(scores, key=scores.get)
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