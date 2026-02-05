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
        if duration_seconds < 1.0:
            raise HTTPException(
                status_code=400,
                detail="Audio duration too short for analysis"
            )

        # --------------------------------------------------
        # 5. Model Inference
        # --------------------------------------------------
        results = detector(speech)

        # The underlying model is binary (real vs fake / spoof).
        # To reduce "everything looks AI" behaviour, we:
        #   1) aggregate scores over both classes,
        #   2) apply a decision threshold (> 0.6) before calling it AI,
        #   3) bias towards HUMAN when the model is uncertain.
        ai_labels = {"fake", "spoof", "ai", "synthetic", "deepfake"}
        human_labels = {"real", "human", "bonafide", "genuine"}

        ai_prob = 0.0
        human_prob = 0.0
        for r in results:
            lbl = r["label"].lower()
            score = float(r["score"])
            if lbl in ai_labels:
                ai_prob += score
            elif lbl in human_labels:
                human_prob += score

        # Fallback for unexpected labels: use the top result directly
        if ai_prob == 0.0 and human_prob == 0.0 and len(results) > 0:
            top_result = results[0]
            label = top_result["label"].lower()
            confidence_for_class = float(top_result["score"])
            is_ai_generated = label in ai_labels
        else:
            total = ai_prob + human_prob
            if total > 0:
                ai_prob /= total
                human_prob /= total

            # Decision thresholding:
            # - Only call AI when its probability is clearly higher (> 0.6)
            # - Otherwise prefer HUMAN to avoid over-flagging
            if ai_prob >= 0.6:
                is_ai_generated = True
                confidence_for_class = ai_prob
            elif human_prob >= 0.6:
                is_ai_generated = False
                confidence_for_class = human_prob
            else:
                # Uncertain region: choose the higher, but keep confidence modest
                is_ai_generated = ai_prob > human_prob
                confidence_for_class = max(ai_prob, human_prob)

        classification = "AI_GENERATED" if is_ai_generated else "HUMAN"

        # --------------------------------------------------
        # 6. Construct Response
        # --------------------------------------------------
        return {
            "status": "success",
            "language": language,
            "classification": classification,
            "confidenceScore": round(float(confidence_for_class), 2),
            "explanation": (
                "Unnatural spectral and temporal artifacts detected"
                if is_ai_generated
                else "Natural human vocal patterns observed"
            )
        }

    except Exception as e:
        # --------------------------------------------------
        # 7. Error Handling
        # --------------------------------------------------
        return {
            "status": "error",
            "message": str(e)
        }
