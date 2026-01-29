"use client";

import React, { useState } from "react";
import DropzoneAreaExample from "./components/dropzone";

interface Prediction {
  rank: number;
  class_name: string;
  class_name_fr: string;
  confidence: number;
  confidence_percent: string;
}

export default function Home() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);

  return (
    <div className="flex justify-center bg-gradient-to-b from-blue-50 via-indigo-50 to-purple-50 min-h-screen py-16 px-4">
      <div className="w-full max-w-4xl">
        {/* Title */}
        <h1 className="text-5xl font-extrabold text-center text-indigo-800 mb-12 tracking-wide drop-shadow-lg">
          BIRD MAN
        </h1>

        {/* Dropzone */}
        <div className="mb-12">
          <DropzoneAreaExample
            setPredictions={setPredictions}
            setLoading={setLoading}
          />
        </div>

        {/* Three prediction boxes */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
          {predictions.length > 0 ? (
            predictions.map((p) => (
              <div
                key={p.rank}
                className={`p-6 rounded-xl shadow-lg flex flex-col items-center justify-center hover:scale-105 transition-transform duration-300 ${
                  p.rank === 1
                    ? "bg-indigo-100 text-indigo-800"
                    : p.rank === 2
                    ? "bg-pink-100 text-pink-800"
                    : "bg-green-100 text-green-800"
                }`}
              >
                <h2 className="text-xl font-semibold mb-2">
                  {p.rank}. {p.class_name_fr}
                </h2>
                <p className="text-center">
                  {p.class_name} <br />
                  Confiance : {p.confidence_percent}
                </p>
              </div>
            ))
          ) : (
            <>
              <div className="bg-indigo-100 p-6 rounded-xl shadow-lg flex flex-col items-center justify-center">
                <h2 className="text-xl font-semibold mb-2">Feature 1</h2>
                <p className="text-center text-indigo-700">
                  Les prédictions apparaîtront ici après l'upload
                </p>
              </div>
              <div className="bg-pink-100 p-6 rounded-xl shadow-lg flex flex-col items-center justify-center">
                <h2 className="text-xl font-semibold mb-2">Feature 2</h2>
                <p className="text-center text-pink-700">
                  Les prédictions apparaîtront ici après l'upload
                </p>
              </div>
              <div className="bg-green-100 p-6 rounded-xl shadow-lg flex flex-col items-center justify-center">
                <h2 className="text-xl font-semibold mb-2">Feature 3</h2>
                <p className="text-center text-green-700">
                  Les prédictions apparaîtront ici après l'upload
                </p>
              </div>
            </>
          )}
        </div>

        {/* About section */}
        <div className="bg-white p-12 rounded-2xl shadow-2xl text-center">
          <h2 className="text-3xl font-bold text-indigo-900 mb-6">
            About BIRD MAN
          </h2>
          <p className="text-indigo-700 text-lg leading-relaxed">
            BIRD MAN est une plateforme professionnelle pour classifier les
            oiseaux. Téléversez une image et obtenez les espèces avec leur
            niveau de confiance en quelques secondes. Expérience simple, rapide
            et élégante.
          </p>
          {loading && (
            <p className="mt-4 text-indigo-600 font-medium">Analyse en cours...</p>
          )}
        </div>
      </div>
    </div>
  );
}