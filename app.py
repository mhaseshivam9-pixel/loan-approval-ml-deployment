
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "loan_approval_deployment_model.joblib"

app = FastAPI(
    title="Loan Approval Prediction API",
    description="API for predicting loan approval",
    version="1.0.0",
)

# Allow requests from a separate frontend.
# For production, replace "*" with your actual website domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_error = None


def load_model():
    """Load the trained model from the repository's model folder."""
    global model, model_error

    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model file not found at: {MODEL_PATH}"
            )

        model = joblib.load(MODEL_PATH)
        model_error = None
        print(f"Model loaded successfully: {MODEL_PATH}")

    except Exception as exc:
        model = None
        model_error = str(exc)
        print(f"MODEL LOADING ERROR: {exc}")


# Attempt loading during application startup.
load_model()


@app.get("/")
def home():
    return {
        "message": "Loan Approval Prediction API is running",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict",
    }


@app.get("/health")
def health():
    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "detail": model_error or "Model is unavailable",
            },
        )

    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(payload: dict):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is unavailable: {model_error}",
        )

    # Accept either {"feature1": value, ...}
    # or {"data": {"feature1": value, ...}}
    features = payload.get("data", payload)

    if not isinstance(features, dict) or not features:
        raise HTTPException(
            status_code=422,
            detail="Send a non-empty JSON object containing model features.",
        )

    try:
        # Preserve the feature order expected by the trained model.
        expected_features = getattr(model, "feature_names_in_", None)

        if expected_features is not None:
            missing = [
                name for name in expected_features
                if name not in features
            ]

            if missing:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": "Required model features are missing.",
                        "missing_features": missing,
                    },
                )

            input_df = pd.DataFrame(
                [[features[name] for name in expected_features]],
                columns=list(expected_features),
            )
        else:
            input_df = pd.DataFrame([features])

        prediction = model.predict(input_df)[0]

        if isinstance(prediction, np.generic):
            prediction = prediction.item()

        result = {
            "prediction": prediction,
        }

        # Include class probabilities when the model supports them.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = getattr(model, "classes_", range(len(probabilities)))

            result["probabilities"] = {
                str(label): float(probability)
                for label, probability in zip(classes, probabilities)
            }

        return result

    except HTTPException:
        raise
    except Exception as exc:
        print(f"PREDICTION ERROR: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )
