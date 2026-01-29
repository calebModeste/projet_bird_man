// Exemple d'utilisation de l'API Bird Classification avec React/Next.js
// =====================================================================

// Types TypeScript
interface Prediction {
  rank: number;
  class_name: string;
  class_name_fr: string;
  confidence: number;
  confidence_percent: string;
}

interface PredictionResponse {
  success: boolean;
  message: string;
  predictions: Prediction[];
  top_prediction: Prediction | null;
}

// Configuration de l'API
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// =====================================================================
// MÉTHODE 1 : Upload de fichier (FormData)
// =====================================================================

export async function classifyBirdImage(
  file: File,
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Erreur API: ${response.status}`);
  }

  return response.json();
}

// =====================================================================
// MÉTHODE 2 : Image en Base64
// =====================================================================

export async function classifyBirdImageBase64(
  imageBase64: string,
): Promise<PredictionResponse> {
  const response = await fetch(`${API_URL}/predict/base64`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ image: imageBase64 }),
  });

  if (!response.ok) {
    throw new Error(`Erreur API: ${response.status}`);
  }

  return response.json();
}

// =====================================================================
// MÉTHODE 3 : Récupérer la liste des classes
// =====================================================================

export async function getBirdClasses(): Promise<{
  count: number;
  classes: Array<{ id: number; name: string; name_fr: string }>;
}> {
  const response = await fetch(`${API_URL}/classes`);
  return response.json();
}

// =====================================================================
// EXEMPLE DE COMPOSANT REACT/NEXT.JS
// =====================================================================

/*
'use client';

import { useState } from 'react';
import { classifyBirdImage, Prediction } from '@/lib/api';

export default function BirdClassifier() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Afficher l'aperçu
    setPreview(URL.createObjectURL(file));
    setLoading(true);

    try {
      const result = await classifyBirdImage(file);
      setPredictions(result.predictions);
    } catch (error) {
      console.error('Erreur:', error);
      alert('Erreur lors de la classification');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">🐦 Classification d'Oiseaux</h1>
      
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="mb-4 p-2 border rounded"
      />

      {preview && (
        <img src={preview} alt="Preview" className="max-w-md mb-4 rounded-lg shadow" />
      )}

      {loading && <p className="text-blue-500">Classification en cours...</p>}

      {predictions.length > 0 && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold mb-4">Résultats :</h2>
          <div className="space-y-3">
            {predictions.map((pred) => (
              <div
                key={pred.rank}
                className={`p-4 rounded-lg ${
                  pred.rank === 1 ? 'bg-green-100 border-2 border-green-500' : 'bg-gray-100'
                }`}
              >
                <div className="flex justify-between items-center">
                  <div>
                    <span className="font-bold text-lg">#{pred.rank}</span>
                    <span className="ml-3 text-lg">{pred.class_name_fr}</span>
                    <span className="ml-2 text-gray-500">({pred.class_name})</span>
                  </div>
                  <span className={`font-bold ${pred.rank === 1 ? 'text-green-600' : 'text-gray-600'}`}>
                    {pred.confidence_percent}
                  </span>
                </div>
                <div className="mt-2 bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${pred.rank === 1 ? 'bg-green-500' : 'bg-blue-400'}`}
                    style={{ width: `${pred.confidence * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
*/
