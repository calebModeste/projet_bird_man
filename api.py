"""
Bird Classification API (PyTorch - ResNet18)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import json
import io
import os

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
MODEL_PATH = "best_resnet18.pt"
CLASS_NAMES_PATH = "class_names.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD CLASS NAMES
# =========================
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)

# =========================
# FASTAPI INIT
# =========================
app = FastAPI(title="Bird Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# RESPONSE MODELS
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
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✓ Model loaded")
print("✓ Classes:", NUM_CLASSES)

# =========================
# PREPROCESS (MUST MATCH TRAINING)
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def preprocess(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image).unsqueeze(0)

# =========================
# ROUTE
# =========================
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(400, "Unsupported file type")

    image = Image.open(io.BytesIO(await file.read()))
    x = preprocess(image).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_idx = torch.topk(probs, 3).indices.tolist()

    predictions = []
    for rank, idx in enumerate(top_idx, 1):
        p = probs[idx].item()
        predictions.append(
            Prediction(
                rank=rank,
                class_name=CLASS_NAMES[idx],
                confidence=p,
                confidence_percent=f"{p*100:.2f}%"
            )
        )

    return PredictionResponse(
        success=True,
        predictions=predictions,
        top_prediction=predictions[0]
    )

# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
