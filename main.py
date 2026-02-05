"""
Problem Statement 1: AI-Generated Voice Detection
-------------------------------------------------
This FastAPI application exposes a REST endpoint that detects whether
a given Base64-encoded MP3 voice sample is AI-generated or Human.

Optimized for Railway (1GB RAM) and high-accuracy detection.
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
@app.post("/")
async def root():
    return {
        "status": "success", 
        "message": "AI Voice Detection API is Online",
        "usage": "POST to /api/voice-detection with audioBase64"
    }

# --------------------------------------------------
# Configuration
# --------------------------------------------------

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
VALID_API_KEY = os.getenv("API_KEY", "guvi-hcl-voice-ai-2026")
DEVICE = 0 if torch.cuda.is_available() else -1

# --------------------------------------------------
# Model Loading (Loaded once at startup)
# --------------------------------------------------

# Using the specialized Deepfake detection model
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
    # 1. API Key Authentication
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    # 2. Input Validation
    language = payload.get("language")
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language. Supported: {SUPPORTED_LANGUAGES}")

    if payload.get("audioFormat") != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format is supported")

    audio_b64 = payload.get("audioBase64")
    if not audio_b64:
        raise HTTPException(status_code=400, detail="audioBase64 field missing")

    try:
        # 3. Decode Base64 Audio
        audio_bytes = base64.b64decode(audio_b64)
        temp_file = f"/tmp/{uuid.uuid4()}.mp3"
        
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)

        # 4. Optimized Audio Loading (Fixes Timeouts & Fuzzy Seeking)
        # We load a 4-second window to ensure fast processing on Railway CPUs.
        # We skip the first 0.5s to avoid MP3 header artifacts.
        speech, sr = librosa.load(temp_file, sr=16000, offset=0.5, duration=4.0)

        # 5. Preprocessing for Accuracy (Stops False AI Flags)
        # Trim silence and normalize volume
        speech, _ = librosa.effects.trim(speech)
        if len(speech) == 0:
            raise ValueError("Audio is silent or contains no speech signal")
            
        speech = librosa.util.normalize(speech)

        # 6. Model Inference
        results = detector(speech)
        
        # 7. Calibrated Classification Logic
        # Extract individual scores for manual thresholding
        ai_score = 0.0
        human_score = 0.0
        
        for res in results:
            label_lower = res["label"].lower()
            if label_lower in ["fake", "spoof", "ai", "synthetic", "label_1"]:
                ai_score = res["score"]
            else:
                human_score = res["score"]

        # 8. Threshold Tuning
        # We only mark as AI if the model is VERY certain (85%+).
        # This prevents natural Indian accents from being misclassified.
        if ai_score > 0.85:
            classification = "AI_GENERATED"
            final_confidence = ai_score
            explanation = "Unnatural spectral and temporal artifacts detected"
        else:
            classification = "HUMAN"
            final_confidence = human_score
            explanation = "Natural human vocal patterns observed"

        # 9. Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

        return {
            "status": "success",
            "language": language,
            "classification": classification,
            "confidenceScore": round(min(float(final_confidence), 0.98), 2),
            "explanation": explanation
        }

    except Exception as e:
        # Clean up if error occurs
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
            
        return {
            "status": "error",
            "message": str(e)
        }
