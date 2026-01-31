# Bird Classification Project

Ce projet vise à développer un modèle de classification d'oiseaux à l'aide du deep learning.

## Description

Le projet utilise des techniques de machine learning et deep learning pour classifier différentes espèces d'oiseaux indiens. Le modèle est entraîné sur un dataset contenant 24 espèces d'oiseaux différentes avec des images d'entraînement et de validation.

## Espèces d'oiseaux

1. Asian-Green-Bee-Eater
2. Brown-Headed-Barbet
3. Cattle-Egret
4. Common-Kingfisher
5. Common-Myna
6. Common-Rosefinch
7. Common-Tailorbird
8. Coppersmith-Barbet
9. Forest-Wagtail
10. Gray-Wagtail
11. Hoopoe
12. House-Crow
13. Indian-Grey-Hornbill
14. Indian-Peacock
15. Indian-Pitta
16. Indian-Roller
17. Jungle-Babbler
18. Northern-Lapwing
19. Red-Wattled-Lapwing
20. Ruddy-Shelduck
21. Rufous-Treepie
22. Sarus-Crane
23. White-Breasted-Kingfisher
24. White-Breasted-Waterhen
25. White-Wagtail

## Structure du projet

```
projet_birdMan/
├── projet_birdMan.ipynb          # Notebook principal (Transfer Learning MobileNetV2)
├── projet_birdMan_CNN.ipynb      # Notebook CNN classique (sans Transfer Learning)
├── api.py                        # API FastAPI pour la classification
├── requirements-api.txt          # Dépendances de l'API
├── example-nextjs-usage.ts       # Exemple d'utilisation React/Next.js
├── README.md                     # Ce fichier
├── data/
│   ├── train_bird/               # Données d'entraînement (25 espèces)
│   └── valid_bird/               # Données de validation (25 espèces)
```

## Architecture des branches

### Branche `main`

Contient l'API complète:

- **API PyTorch + ResNet18** : Modèle entraîné avec PyTorch (`best_resnet18.pt`)
- **fichier** `projet_birdMan.ipynb`

### Branche `developp`

Contient l'API complète:

- **API TensorFlow + MobileNetV2** : Modèle fine-tuné MobileNetV2 (`best_model_mobilenet_finetuned.h5`)
- **fichier** `projet_birdManTran.ipynb`

### Branche `next_projet`

Contient l'interface frontend React/Next.js pour interagir avec l'API

## Déploiement

L'API est actuellement déployée sur **[Render](https://render.com)** :

**URL de l'API déployée :** [https://projet-bird-man-1.onrender.com/docs](https://projet-bird-man-1.onrender.com/docs)

> **Attention** : Ce déploiement est temporaire.

## Dataset

> **Source du dataset :** [Lien Google Drive](https://drive.google.com/drive/folders/1kHTcb7OktpYB9vUaZPLQ3ywXFYMUdQsP?usp=sharing) || [Lien Kaggle](https://www.kaggle.com/code/hamedghorbani/25-indian-bird-specie-image-classification-98-5)

## Requêtes

Les packages suivants sont nécessaires pour exécuter ce projet :

- TensorFlow / PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Utilisation

Ouvrez et exécutez le notebook `projet_birdMan.ipynb` dans Jupyter Notebook ou JupyterLab.

## API de Classification

Une API REST est disponible pour classifier des images d'oiseaux depuis une application web (React/Next.js).

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Lancer l'API

```bash
# Mode développement (avec auto-reload)
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Ou directement avec Python
python api.py
```

### Endpoints disponibles

| Méthode | Endpoint          | Description                     |
| ------- | ----------------- | ------------------------------- |
| `GET`   | `/`               | Page d'accueil                  |
| `GET`   | `/health`         | État de l'API                   |
| `GET`   | `/classes`        | Liste des 25 espèces d'oiseaux  |
| `GET`   | `/docs`           | Documentation Swagger           |
| `POST`  | `/predict`        | Classifier une image (FormData) |
| `POST`  | `/predict/base64` | Classifier une image (Base64)   |

### Exemple de réponse

```json
{
  "success": true,
  "message": "Classification réussie",
  "predictions": [
    {
      "rank": 1,
      "class_name": "Indian-Peacock",
      "class_name_fr": "Paon bleu",
      "confidence": 0.89
    },
    {
      "rank": 2,
      "class_name": "Indian-Roller",
      "class_name_fr": "Rollier indien",
      "confidence": 0.06
    },
    {
      "rank": 3,
      "class_name": "Common-Kingfisher",
      "class_name_fr": "Martin-pêcheur",
      "confidence": 0.03
    }
  ],
  "top_prediction": {
    "rank": 1,
    "class_name": "Indian-Peacock",
    "confidence": 0.89
  }
}
```

### Utilisation avec React/Next.js

```typescript
const formData = new FormData();
formData.append("file", imageFile);

const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
});

const result = await response.json();
console.log(result.predictions); // Top 3 prédictions
```

## Auteur

> - [Akram CHAMI](https://github.com/akramchami99)
> - [Caleb Nathanael TSIBA MODESTE](https://github.com/calebModeste)
> - [James MBA FONGANG](https://github.com/MFJD)

Master IA - Cours de Machine et Deep Learning

<!-- ## Licence

À définir -->
