"use client";

import React, { useState } from "react";
import { DropzoneArea } from "mui-file-dropzone";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";

interface Prediction {
  rank: number;
  class_name: string;
  class_name_fr: string;
  confidence: number;
  confidence_percent: string;
}

interface DropzoneProps {
  setPredictions: React.Dispatch<React.SetStateAction<Prediction[]>>;
  setLoading: React.Dispatch<React.SetStateAction<boolean>>;
}

const DropzoneAreaExample: React.FC<DropzoneProps> = ({
  setPredictions,
  setLoading,
}) => {
  const [files, setFiles] = useState<File[]>([]);

  const handleUpload = async (file: File) => {
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("https://projet-bird-man-1.onrender.com/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (data.success) {
        setPredictions(data.predictions);
      } else {
        alert("Erreur lors de la classification");
      }
    } catch (error) {
      console.error(error);
      alert("Erreur lors de l'appel à l'API");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative p-4 rounded-xl shadow-lg border-2 border-dashed border-indigo-300 bg-white hover:shadow-xl transition-all duration-300 max-w-md mx-auto">
      <div className="flex flex-col items-center justify-center mb-4">
        <CloudUploadIcon className="text-indigo-500 text-5xl mb-2 animate-bounce" />
        <h2 className="text-xl font-bold text-indigo-800">Upload Image</h2>
        <p className="text-indigo-600 text-center mt-1 text-sm">
          Glissez & déposez ou cliquez pour sélectionner une image
        </p>
      </div>

      <DropzoneArea
        onChange={(newFiles) => {
          setFiles(newFiles);
          if (newFiles.length > 0) handleUpload(newFiles[0]);
        }}
        showPreviews={true}
        showPreviewsInDropzone={true}
        filesLimit={1}
        acceptedFiles={["image/*"]}
        dropzoneText=""
        classes={{
          root: "bg-white rounded-xl p-2 shadow-md hover:shadow-lg hover:bg-indigo-50 transition-all duration-300 border border-dashed border-indigo-200",
        }}
      />
    </div>
  );
};

export default DropzoneAreaExample;
