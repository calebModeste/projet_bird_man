"""
API de Classification d'Oiseaux - Bird Classification API
=========================================================
API REST utilisant FastAPI pour classifier des images d'oiseaux
Support des modèles TensorFlow/Keras ET PyTorch

Usage:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Literal
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = 224

# Chemins des modèles
TENSORFLOW_MODEL_PATHS = [
    "best_model_mobilenet_finetuned.h5",
    "best_model_mobilenet.h5",
    "bird_classifier_mobilenet.h5"
]

PYTORCH_MODEL_PATHS = [
    "best_model_pytorch.pth",
    "bird_classifier_pytorch.pth",
    "model_pytorch.pt"
]

# Framework actif (sera détecté automatiquement)
ACTIVE_FRAMEWORK: Optional[Literal["tensorflow", "pytorch"]] = None

# Liste des classes d'oiseaux
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

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Bird Classification API",
    description="API pour classifier des images d'oiseaux - Support TensorFlow & PyTorch",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
model = None
model_path_loaded = None

# ============================================================================
# MODÈLES PYDANTIC
# ============================================================================

class Prediction(BaseModel):
    rank: int
    class_name: str
    class_name_fr: str
    confidence: float
    confidence_percent: str


class PredictionResponse(BaseModel):
    success: bool
    message: str
    framework: Optional[str] = None
    predictions: List[Prediction]
    top_prediction: Optional[Prediction] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    framework: Optional[str] = None
    model_path: Optional[str] = None
    num_classes: int
    supported_frameworks: List[str]


# ============================================================================
# CHARGEMENT DES MODÈLES
# ============================================================================

def load_tensorflow_model() -> tuple:
    """Charge un modèle TensorFlow/Keras"""
    try:
        import tensorflow as tf
        
        for model_path in TENSORFLOW_MODEL_PATHS:
            if os.path.exists(model_path):
                loaded_model = tf.keras.models.load_model(model_path)
                print(f"✓ Modèle TensorFlow chargé : {model_path}")
                return loaded_model, model_path, "tensorflow"
        
        return None, None, None
        
    except ImportError:
        print("⚠ TensorFlow non installé")
        return None, None, None
    except Exception as e:
        print(f"❌ Erreur TensorFlow : {e}")
        return None, None, None


def load_pytorch_model() -> tuple:
    """Charge un modèle PyTorch"""
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
        
        # Définir l'architecture du modèle (MobileNetV2 modifié)
        class BirdClassifierPyTorch(nn.Module):
            def __init__(self, num_classes=25):
                super(BirdClassifierPyTorch, self).__init__()
                # Charger MobileNetV2 pré-entraîné
                self.base_model = models.mobilenet_v2(pretrained=False)
                
                # Remplacer le classificateur
                in_features = self.base_model.classifier[1].in_features
                self.base_model.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_features, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_classes)
                )
            
            def forward(self, x):
                return self.base_model(x)
        
        # Chercher un modèle PyTorch
        for model_path in PYTORCH_MODEL_PATHS:
            if os.path.exists(model_path):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Essayer de charger comme state_dict
                try:
                    loaded_model = BirdClassifierPyTorch(num_classes=len(CLASS_NAMES))
                    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
                    loaded_model.to(device)
                    loaded_model.eval()
                    print(f"✓ Modèle PyTorch chargé (state_dict) : {model_path}")
                    return loaded_model, model_path, "pytorch"
                except:
                    # Essayer de charger le modèle complet
                    try:
                        loaded_model = torch.load(model_path, map_location=device)
                        loaded_model.to(device)
                        loaded_model.eval()
                        print(f"✓ Modèle PyTorch chargé (complet) : {model_path}")
                        return loaded_model, model_path, "pytorch"
                    except Exception as e:
                        print(f"⚠ Impossible de charger {model_path}: {e}")
                        continue
        
        return None, None, None
        
    except ImportError:
        print("⚠ PyTorch non installé")
        return None, None, None
    except Exception as e:
        print(f"❌ Erreur PyTorch : {e}")
        return None, None, None


def load_model():
    """Charge le modèle disponible (TensorFlow ou PyTorch)"""
    global model, model_path_loaded, ACTIVE_FRAMEWORK
    
    print("\n🔍 Recherche de modèles...")
    
    # Essayer TensorFlow d'abord
    tf_model, tf_path, tf_framework = load_tensorflow_model()
    if tf_model is not None:
        model = tf_model
        model_path_loaded = tf_path
        ACTIVE_FRAMEWORK = tf_framework
        return tf_path
    
    # Sinon essayer PyTorch
    pt_model, pt_path, pt_framework = load_pytorch_model()
    if pt_model is not None:
        model = pt_model
        model_path_loaded = pt_path
        ACTIVE_FRAMEWORK = pt_framework
        return pt_path
    
    print("⚠ Aucun modèle trouvé. L'API fonctionnera en mode demo.")
    ACTIVE_FRAMEWORK = None
    return None


# ============================================================================
# PRÉTRAITEMENT DES IMAGES
# ============================================================================

def preprocess_image_tensorflow(image: Image.Image) -> np.ndarray:
    """Prétraitement pour TensorFlow"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def preprocess_image_pytorch(image: Image.Image):
    """Prétraitement pour PyTorch"""
    import torch
    from torchvision import transforms
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Transformations standard pour MobileNetV2
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)  # Ajouter batch dimension
    
    return img_tensor


def preprocess_image(image: Image.Image):
    """Prétraitement adapté au framework actif"""
    if ACTIVE_FRAMEWORK == "pytorch":
        return preprocess_image_pytorch(image)
    else:
        return preprocess_image_tensorflow(image)


# ============================================================================
# PRÉDICTION
# ============================================================================

def predict_tensorflow(processed_image: np.ndarray) -> np.ndarray:
    """Prédiction avec TensorFlow"""
    predictions = model.predict(processed_image, verbose=0)
    return predictions


def predict_pytorch(processed_image) -> np.ndarray:
    """Prédiction avec PyTorch"""
    import torch
    import torch.nn.functional as F
    
    device = next(model.parameters()).device
    processed_image = processed_image.to(device)
    
    with torch.no_grad():
        outputs = model(processed_image)
        probabilities = F.softmax(outputs, dim=1)
        predictions = probabilities.cpu().numpy()
    
    return predictions


def make_prediction(processed_image) -> np.ndarray:
    """Fait une prédiction avec le framework actif"""
    if model is None:
        # Mode demo
        predictions = np.random.rand(1, len(CLASS_NAMES))
        predictions = predictions / predictions.sum()
        return predictions
    
    if ACTIVE_FRAMEWORK == "pytorch":
        return predict_pytorch(processed_image)
    else:
        return predict_tensorflow(processed_image)


def get_top_predictions(predictions: np.ndarray, top_k: int = 3) -> List[Prediction]:
    """Récupère les top K prédictions"""
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


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Événement de démarrage"""
    print("\n" + "="*60)
    print("🐦 Bird Classification API - Démarrage")
    print("="*60)
    load_model()
    print(f"📊 Nombre de classes : {len(CLASS_NAMES)}")
    print(f"📐 Taille des images : {IMG_SIZE}x{IMG_SIZE}")
    print(f"🔧 Framework actif : {ACTIVE_FRAMEWORK or 'Demo mode'}")
    print("="*60 + "\n")


@app.get("/", response_model=dict)
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "🐦 Bird Classification API",
        "version": "2.0.0",
        "framework": ACTIVE_FRAMEWORK or "demo",
        "endpoints": {
            "POST /predict": "Classifier une image d'oiseau (upload)",
            "POST /predict/base64": "Classifier une image en base64",
            "GET /health": "Vérifier l'état de l'API",
            "GET /classes": "Liste des classes d'oiseaux",
            "GET /docs": "Documentation Swagger"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de l'état de l'API"""
    supported = []
    try:
        import tensorflow
        supported.append("tensorflow")
    except:
        pass
    try:
        import torch
        supported.append("pytorch")
    except:
        pass
    
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        framework=ACTIVE_FRAMEWORK,
        model_path=model_path_loaded,
        num_classes=len(CLASS_NAMES),
        supported_frameworks=supported
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
    
    Retourne les 3 prédictions les plus probables
    """
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Type non supporté: {file.content_type}. Utilisez JPEG, PNG ou WebP."
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        processed_image = preprocess_image(image)
        predictions = make_prediction(processed_image)
        top_predictions = get_top_predictions(predictions, top_k=3)
        
        return PredictionResponse(
            success=True,
            message="Classification réussie",
            framework=ACTIVE_FRAMEWORK or "demo",
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
        
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        processed_image = preprocess_image(image)
        predictions = make_prediction(processed_image)
        top_predictions = get_top_predictions(predictions, top_k=3)
        
        return PredictionResponse(
            success=True,
            message="Classification réussie",
            framework=ACTIVE_FRAMEWORK or "demo",
            predictions=top_predictions,
            top_prediction=top_predictions[0] if top_predictions else None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la classification : {str(e)}"
        )


@app.post("/reload-model")
async def reload_model_endpoint():
    """Recharge le modèle (utile après un changement de modèle)"""
    global model, model_path_loaded, ACTIVE_FRAMEWORK
    
    model = None
    model_path_loaded = None
    ACTIVE_FRAMEWORK = None
    
    loaded_path = load_model()
    
    return {
        "success": model is not None,
        "framework": ACTIVE_FRAMEWORK,
        "model_path": loaded_path,
        "message": f"Modèle rechargé avec {ACTIVE_FRAMEWORK}" if model else "Aucun modèle trouvé"
    }


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Démarrage du serveur de développement...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
