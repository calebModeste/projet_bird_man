"""
API de Classification d'Oiseaux - Bird Classification API (PyTorch)
==================================================================
API REST utilisant FastAPI pour classifier des images d'oiseaux
avec un modèle ResNet18 entraîné (Transfer Learning)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import io
import os

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
MODEL_PATH = "best_resnet18.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⚠️ MUST match training order
CLASS_NAMES = [
    "Asian-Green-Bee-Eater",
    "Brown-Headed-Barbet",
    "Cattle-Egret",
    "Common-Kingfisher",
    "Common-Myna",
    "Common-Rosefinch",
    "Common-Tailorbird",
    "Coppersmith-Barbet",
    "Forest-Wagtail",
    "Gray-Wagtail",
    "Hoopoe",
    "House-Crow",
    "Indian-Grey-Hornbill",
    "Indian-Peacock",
    "Indian-Pitta",
    "Indian-Roller",
    "Jungle-Babbler",
    "Northern-Lapwing",
    "Red-Wattled-Lapwing",
    "Ruddy-Shelduck",
    "Rufous-Treepie",
    "Sarus-Crane",
    "White-Breasted-Kingfisher",
    "White-Breasted-Waterhen",
    "White-Wagtail"
]

# =========================
# FASTAPI INIT
# =========================
app = FastAPI(
    title="Bird Classification API (PyTorch)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PYDANTIC MODELS
# =========================
class Prediction(BaseModel):
    rank: int
    class_name: str
    confidence: float
    confidence_percent: str


class PredictionResponse(BaseModel):
    success: bool
    predictions: List[Prediction]
    top_prediction: Prediction


# =========================
# LOAD MODEL
# =========================
model = None

def load_model():
    global model

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("✓ Modèle PyTorch chargé :", MODEL_PATH)


# =========================
# PREPROCESS (IMAGENET)
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image).unsqueeze(0)


# =========================
# STARTUP
# =========================
@app.on_event("startup")
async def startup():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("❌ Modèle PyTorch introuvable")
    load_model()


# =========================
# ROUTES
# =========================
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Format non supporté")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    x = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    top_idx = probs.argsort()[::-1][:3]

    predictions = [
        Prediction(
            rank=i + 1,
            class_name=CLASS_NAMES[idx],
            confidence=float(probs[idx]),
            confidence_percent=f"{probs[idx]*100:.2f}%"
        )
        for i, idx in enumerate(top_idx)
    ]

    return PredictionResponse(
        success=True,
        predictions=predictions,
        top_prediction=predictions[0]
    )


# =========================
# RUN (DEV)
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)