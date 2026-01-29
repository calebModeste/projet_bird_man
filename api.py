"""
API de Classification d'Oiseaux - Bird Classification API
=========================================================
API REST utilisant FastAPI pour classifier des images d'oiseaux
avec les modèles entraînés (MobileNetV2 Transfer Learning)

Usage:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path

# Configuration
IMG_SIZE = 224
MODEL_PATH = "best_model_mobilenet_finetuned.h5"
MODEL_PATH_BACKUP = "best_model_mobilenet.h5"

# Liste des classes d'oiseaux (à mettre à jour selon votre dataset)
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

# Initialiser FastAPI
app = FastAPI(
    title="Bird Classification API",
    description="API pour classifier des images d'oiseaux avec Transfer Learning (MobileNetV2)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS pour React/Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Next.js dev
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "https://localhost:3000",
        "*"                           # Pour le développement (à restreindre en production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable globale pour le modèle
model = None


# Modèles Pydantic pour les réponses
class Prediction(BaseModel):
    rank: int
    class_name: str
    class_name_fr: str
    confidence: float
    confidence_percent: str


class PredictionResponse(BaseModel):
    success: bool
    message: str
    predictions: List[Prediction]
    top_prediction: Optional[Prediction] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    num_classes: int


# Dictionnaire de traduction des noms d'oiseaux
BIRD_NAMES_FR = {
    "Asian-Green-Bee-Eater": "Guêpier d'Orient",
    "Brown-Headed-Barbet": "Barbu à tête brune",
    "Cattle-Egret": "Héron garde-bœufs",
    "Common-Kingfisher": "Martin-pêcheur d'Europe",
    "Common-Myna": "Martin triste",
    "Common-Rosefinch": "Roselin cramoisi",
    "Common-Tailorbird": "Couturière à longue queue",
    "Coppersmith-Barbet": "Barbu à plastron rouge",
    "Forest-Wagtail": "Bergeronnette de forêt",
    "Gray-Wagtail": "Bergeronnette des ruisseaux",
    "Hoopoe": "Huppe fasciée",
    "House-Crow": "Corbeau familier",
    "Indian-Grey-Hornbill": "Calao gris",
    "Indian-Peacock": "Paon bleu",
    "Indian-Pitta": "Brève du Bengale",
    "Indian-Roller": "Rollier indien",
    "Jungle-Babbler": "Cratérope de brousse",
    "Northern-Lapwing": "Vanneau huppé",
    "Red-Wattled-Lapwing": "Vanneau indien",
    "Ruddy-Shelduck": "Tadorne casarca",
    "Rufous-Treepie": "Témia vagabonde",
    "Sarus-Crane": "Grue antigone",
    "White-Breasted-Kingfisher": "Martin-chasseur de Smyrne",
    "White-Breasted-Waterhen": "Râle à poitrine blanche",
    "White-Wagtail": "Bergeronnette grise"
}


def load_model():
    """Charge le modèle TensorFlow/Keras"""
    global model
    
    try:
        import tensorflow as tf
        
        # Essayer de charger le modèle fine-tuned d'abord
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Modele charge : {MODEL_PATH}")
            return MODEL_PATH
        elif os.path.exists(MODEL_PATH_BACKUP):
            model = tf.keras.models.load_model(MODEL_PATH_BACKUP)
            print(f"Modele charge : {MODEL_PATH_BACKUP}")
            return MODEL_PATH_BACKUP
        else:
            print("Aucun modele trouve. L'API fonctionnera en mode demo.")
            return None
            
    except Exception as e:
        print(f"Erreur lors du chargement du modele : {e}")
        return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prétraitement de l'image pour le modèle"""
    # Convertir en RGB si nécessaire
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionner
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    
    # Convertir en array et normaliser
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Ajouter la dimension batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def get_top_predictions(predictions: np.ndarray, top_k: int = 3) -> List[Prediction]:
    """Récupère les top K prédictions"""
    # Récupérer les indices des top K prédictions
    top_indices = np.argsort(predictions[0])[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Unknown-{idx}"
        confidence = float(predictions[0][idx])
        
        results.append(Prediction(
            rank=rank,
            class_name=class_name,
            class_name_fr=BIRD_NAMES_FR.get(class_name, class_name),
            confidence=round(confidence, 4),
            confidence_percent=f"{confidence * 100:.2f}%"
        ))
    
    return results


@app.on_event("startup")
async def startup_event():
    """Événement de démarrage - charge le modèle"""
    print("\n" + "="*50)
    print("Bird Classification API - Démarrage")
    print("="*50)
    load_model()
    print(f"Nombre de classes : {len(CLASS_NAMES)}")
    print(f"Taille des images : {IMG_SIZE}x{IMG_SIZE}")
    print("="*50 + "\n")


@app.get("/", response_model=dict)
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "Bird Classification API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Classifier une image d'oiseau",
            "GET /health": "Vérifier l'état de l'API",
            "GET /classes": "Liste des classes d'oiseaux",
            "GET /docs": "Documentation Swagger"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de l'état de l'API"""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_PATH_BACKUP,
        num_classes=len(CLASS_NAMES)
    )


@app.get("/classes", response_model=dict)
async def get_classes():
    """Retourne la liste des classes d'oiseaux"""
    classes = []
    for i, name in enumerate(CLASS_NAMES):
        classes.append({
            "id": i,
            "name": name,
            "name_fr": BIRD_NAMES_FR.get(name, name)
        })
    
    return {
        "count": len(CLASS_NAMES),
        "classes": classes
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Classifier une image d'oiseau
    
    - **file**: Image à classifier (JPEG, PNG, WebP)
    
    Retourne les 3 prédictions les plus probables avec:
    - Rang (1, 2, 3)
    - Nom de l'espèce (EN/FR)
    - Score de confiance
    """
    
    # Vérifier le type de fichier
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Type de fichier non supporté: {file.content_type}. Utilisez JPEG, PNG ou WebP."
        )
    
    try:
        # Lire l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Prétraiter l'image
        processed_image = preprocess_image(image)
        
        # Faire la prédiction
        if model is not None:
            predictions = model.predict(processed_image, verbose=0)
        else:
            # Mode demo : prédictions aléatoires
            predictions = np.random.rand(1, len(CLASS_NAMES))
            predictions = predictions / predictions.sum()  # Normaliser
        
        # Récupérer les top 3 prédictions
        top_predictions = get_top_predictions(predictions, top_k=3)
        
        return PredictionResponse(
            success=True,
            message="Classification réussie",
            predictions=top_predictions,
            top_prediction=top_predictions[0] if top_predictions else None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la classification : {str(e)}"
        )


@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(data: dict):
    """
    Classifier une image en base64
    
    Body JSON:
    ```json
    {
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    }
    ```
    """
    import base64
    
    try:
        image_data = data.get("image", "")
        
        # Extraire les données base64
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        
        # Décoder l'image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Prétraiter l'image
        processed_image = preprocess_image(image)
        
        # Faire la prédiction
        if model is not None:
            predictions = model.predict(processed_image, verbose=0)
        else:
            predictions = np.random.rand(1, len(CLASS_NAMES))
            predictions = predictions / predictions.sum()
        
        # Récupérer les top 3 prédictions
        top_predictions = get_top_predictions(predictions, top_k=3)
        
        return PredictionResponse(
            success=True,
            message="Classification réussie",
            predictions=top_predictions,
            top_prediction=top_predictions[0] if top_predictions else None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la classification : {str(e)}"
        )


# Point d'entrée pour le développement
if __name__ == "__main__":
    import uvicorn
    print("\n Démarrage du serveur de développement...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
