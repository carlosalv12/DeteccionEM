from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

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

# Carga el pipeline UNA sola vez al arrancar
clf = pipeline(
    "text-classification",
    model="carlosalv12/deteccionem-model",
    tokenizer="carlosalv12/deteccionem-model",
    device=-1  # usa CPU; si tu contenedor de Render tiene GPU pon device=0
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o tu dominio
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="El campo 'text' no puede estar vacío")

    # inferencia local
    results = clf(text)
    # results = [{"label": "...", "score": 0.xx}, ...]

    top = results[0]
    label = top["label"]
    # construye dict de scores completo
    scores = {r["label"]: r["score"] for r in results}

    return PredictionResponse(label=label, scores=scores)
