from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    scores: dict

app = FastAPI(
    title="Sistema de detección de enfermedades mentales",
    description="API para la detección de enfermedades mentales a partir de texto",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

HF_TOKEN = os.environ["HF_AUTH_TOKEN"]       # Pon tu token aquí en Render
HF_API = "https://api-inference.huggingface.co/models/carlosalv12/deteccionem-model"

def predict_labels(text: str):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    resp = requests.post(HF_API, headers=headers, json={"inputs": text})
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"HF API error {resp.status_code}")
    out = resp.json()
    top = out[0]
    return top["label"], {item["label"]: item["score"] for item in out}

@app.get("/")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(400, "El campo 'text' no puede estar vacío")
    label, scores = predict_labels(text)
    return PredictionResponse(label=label, scores=scores)
