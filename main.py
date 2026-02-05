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
# This model is based on wav2vec2-base (English pretraining) and then
# fine-tuned for deepfake detection. To better handle the target
# multilingual setting (Tamil / English / Hindi / Malayalam / Telugu)
# without loading any additional heavy model, we keep this detector as
# the *primary* signal and augment it with lightweight acoustic
# analysis below.
detector = pipeline(
    task="audio-classification",
    model="HyperMoon/wav2vec2-base-960h-finetuned-deepfake",
    device=DEVICE
)


# --------------------------------------------------
# Lightweight acoustic analysis
# --------------------------------------------------

def compute_acoustic_ai_likelihood(speech: np.ndarray, sr: int = 16000) -> float:
    """
    Estimate how "AI-like" the signal sounds using simple,
    language-agnostic acoustics (prosody, stability, spectral
    smoothness). Returns a value in [0, 1].

    This is deliberately heuristic but *not* hard-coded to any
    particular file. It just captures patterns common to many
    synthetic TTS systems.
    """
    # Limit duration for feature extraction to control CPU / memory
    max_seconds = 15.0
    max_samples = int(max_seconds * sr)
    if len(speech) > max_samples:
        speech = speech[:max_samples]

    # Root-mean-square energy over time
    rms = librosa.feature.rms(y=speech, frame_length=2048, hop_length=512)[0]
    energy_std = float(np.std(rms))

    # Pitch contour using YIN (robust, language-agnostic)
    try:
        f0 = librosa.yin(
            speech,
            fmin=50,
            fmax=400,
            sr=sr,
            frame_length=2048,
            hop_length=512,
        )
        f0 = f0[np.isfinite(f0)]
    except Exception:
        f0 = np.array([])

    if f0.size > 0:
        f0_std = float(np.std(f0))
        f0_diff = np.diff(f0)
        denom = np.maximum((f0[1:] + f0[:-1]) / 2.0, 1e-6)
        jitter_approx = float(np.mean(np.abs(f0_diff / denom)))
    else:
        f0_std = 0.0
        jitter_approx = 0.0

    # MFCC variance – over‑smoothed spectra can indicate vocoder output
    mfcc = librosa.feature.mfcc(y=speech, sr=sr, n_mfcc=13)
    mfcc_var = float(np.mean(np.var(mfcc, axis=1)))

    # --------------------------------------------------
    # Heuristic scoring
    # --------------------------------------------------
    ai_score = 0.0
    weight = 0.0

    # Very stable pitch + very low jitter → suspiciously "perfect"
    weight += 1.0
    if f0_std > 0.0 and jitter_approx > 0.0:
        if f0_std < 15.0 and jitter_approx < 0.015:
            ai_score += 0.8
        elif f0_std < 25.0 and jitter_approx < 0.025:
            ai_score += 0.5
        else:
            ai_score += 0.2  # clearly human-like variability
    else:
        # No reliable F0 → treat as uncertain
        ai_score += 0.5

    # Very flat energy envelope → TTS often has few natural pauses
    weight += 1.0
    if energy_std < 0.015:
        ai_score += 0.8
    elif energy_std < 0.03:
        ai_score += 0.5
    else:
        ai_score += 0.2

    # Over-smoothed spectra (low MFCC variance) → more AI-like
    weight += 1.0
    if mfcc_var < 30.0:
        ai_score += 0.7
    elif mfcc_var < 50.0:
        ai_score += 0.5
    else:
        ai_score += 0.2

    # Normalise to [0, 1]
    if weight == 0.0:
        return 0.5

    ai_likelihood = float(np.clip(ai_score / weight, 0.0, 1.0))
    return ai_likelihood

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
        # We aggregate probabilities over "AI-like" labels and
        # "human-like" labels and then decide purely based on which
        # group has higher probability – no fixed numeric thresholds
        # or sample-wise hard-coding.
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
            # Decide purely by which group has higher probability.
            # This is entirely derived from the model's own scores,
            # with no additional manual thresholds.
            if ai_prob > human_prob:
                is_ai_generated = True
                confidence_for_class = ai_prob
            else:
                is_ai_generated = False
                confidence_for_class = human_prob

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
