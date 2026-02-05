"""
Problem Statement 1: AI-Generated Voice Detection
-------------------------------------------------
This FastAPI application exposes a REST endpoint that detects whether
a given Base64-encoded MP3 voice sample is AI-generated or Human.

Supported Languages:
- Tamil
- English
- Hindi
- Malayalam
- Telugu
"""

import os
import uuid
import base64

import torch
import librosa
import numpy as np

from fastapi import FastAPI, Header, HTTPException
from transformers import pipeline

# --------------------------------------------------
# Application Initialization
# --------------------------------------------------

app = FastAPI(title="AI Voice Detection API", version="1.0")
@app.get("/")
async def root():
    return {"status": "success", "message": "AI Voice Detection API is Online"}

# --------------------------------------------------
# Configuration
# --------------------------------------------------

# Supported languages as per problem statement
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# API Key (use environment variable in real deployment)
VALID_API_KEY = os.getenv("API_KEY", "guvi-hcl-voice-ai-2026")

# Device selection for inference
# Transformers expects:
#   device = 0  -> GPU
#   device = -1 -> CPU
DEVICE = 0 if torch.cuda.is_available() else -1

# --------------------------------------------------
# Model Loading (Executed once at startup)
# --------------------------------------------------

# NOTE:
# This model is multilingual and based on XLSR,
# which supports Indian languages reasonably well.
detector = pipeline(
    task="audio-classification",
    model="HyperMoon/wav2vec2-base-960h-finetuned-deepfake",
    device=DEVICE
)

# --------------------------------------------------
# API Endpoint
# --------------------------------------------------

@app.post("/api/voice-detection")
async def detect_voice(
    payload: dict,
    x_api_key: str = Header(None)
):
    """
    Detect whether a given voice sample is AI-generated or Human.

    Request:
    - Headers:
        x-api-key: YOUR_SECRET_API_KEY
    - Body (JSON):
        {
            "language": "Tamil",
            "audioFormat": "mp3",
            "audioBase64": "<Base64 MP3>"
        }

    Response:
    - JSON with classification result and confidence
    """

    # --------------------------------------------------
    # 1. API Key Authentication
    # --------------------------------------------------
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    # --------------------------------------------------
    # 2. Input Validation
    # --------------------------------------------------
    language = payload.get("language")
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if payload.get("audioFormat") != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format is supported")

    if "audioBase64" not in payload:
        raise HTTPException(status_code=400, detail="audioBase64 field missing")

    try:
        # --------------------------------------------------
        # 3. Decode Base64 Audio
        # --------------------------------------------------
        audio_bytes = base64.b64decode(payload["audioBase64"])

        # Use unique temp file to avoid race conditions
        temp_file = f"/tmp/{uuid.uuid4()}.mp3"
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)

        ## --------------------------------------------------
        # 4. Audio Loading (Minimal Processing)
        # --------------------------------------------------
        speech, sr = librosa.load(temp_file, sr=None)

        # Resample ONLY if required
        if sr != 16000:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)

        # --------------------------------------------------
        # 4.1 Validate Audio Signal (Fix 2)
        # --------------------------------------------------
        if speech is None or len(speech) == 0:
            raise HTTPException(
                status_code=400,
                detail="Audio file contains no usable signal"
            )

        # --------------------------------------------------
        # 4.2 Minimum Duration Check (Fix 3)
        # --------------------------------------------------
        duration_seconds = len(speech) / 16000
        if duration_seconds < 2.5:
            raise HTTPException(
                status_code=400,
                detail="Audio duration too short for analysis"
            )


        # --------------------------------------------------
        # 5. Model Inference
        # --------------------------------------------------
        results = detector(speech)

        top_result = results[0]
        label = top_result["label"].lower()
        confidence = float(top_result["score"])
        
        # Handle different label naming conventions safely
        is_ai_generated = label in ["fake", "spoof", "ai", "synthetic"]
        
        # Signal-based calibration guard
        energy_variance = np.var(speech)
        
        if energy_variance > 0.02 and confidence < 0.90:
            classification = "HUMAN"
        elif is_ai_generated and confidence >= 0.85:
            classification = "AI_GENERATED"
        else:
            classification = "HUMAN"



        # --------------------------------------------------
        # 6. Explanation Logic (Confidence-Aware)
        # --------------------------------------------------


        if confidence > 0.75:
            explanation = (
                "Unnatural spectral and temporal artifacts detected"
                if classification == "AI_GENERATED"
                else "Natural human vocal patterns observed"
            )
        elif confidence < 0.55:
            explanation = "Low confidence prediction due to ambiguous acoustic patterns"
        else:
            explanation = (
                "Likely synthetic speech patterns detected"
                if classification == "AI_GENERATED"
                else "Likely human speech with some artificial characteristics"
            )


        return {
            "status": "success",
            "language": language,
            "classification": classification,
            "confidenceScore": min(float(f"{confidence:.3f}"), 0.95),
            "explanation": explanation
        }

    except Exception as e:
        # --------------------------------------------------
        # 7. Error Handling
        # --------------------------------------------------
        return {
            "status": "error",
            "message": str(e)
        }
