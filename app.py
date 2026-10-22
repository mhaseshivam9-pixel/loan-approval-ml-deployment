from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR / "model" / "loan_approval_deployment_model.joblib"
)

app = FastAPI(
    title="Loan Approval Prediction API",
    description="API for predicting loan approval",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and deployment metadata
model = None
model_metadata = {}
model_error = None


def load_model():
    """Load the estimator from the saved deployment package."""
    global model, model_metadata, model_error

    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model file not found at: {MODEL_PATH}"
            )

        loaded_object = joblib.load(MODEL_PATH)

        if isinstance(loaded_object, dict):
            # The notebook saves the estimator inside this dictionary.
            loaded_model = loaded_object.get("model")

            if loaded_model is None:
                raise ValueError(
                    "The deployment package has no 'model' entry."
                )

            model_metadata = {
                key: value
                for key, value in loaded_object.items()
                if key != "model"
            }
        else:
            # Also support a file containing the estimator directly.
            loaded_model = loaded_object
            model_metadata = {}

        if not callable(getattr(loaded_model, "predict", None)):
            raise ValueError(
                "The loaded object does not provide a predict() method."
            )

        model = loaded_model
        model_error = None

        print(f"Model loaded successfully: {MODEL_PATH}")
        print(f"Loaded estimator type: {type(model).__name__}")
        print(f"Model metadata keys: {list(model_metadata.keys())}")

    except Exception as exc:
        model = None
        model_metadata = {}
        model_error = str(exc)
        print(f"MODEL LOADING ERROR: {exc}")


# Load model when the application starts.
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
        "model_type": type(model).__name__,
    }


@app.post("/predict")
def predict(payload: dict):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is unavailable: {model_error}",
        )

    # Accept either a direct feature object or {"data": {...}}.
    features = payload.get("data", payload)

    if not isinstance(features, dict) or not features:
        raise HTTPException(
            status_code=422,
            detail="Send a non-empty JSON object containing model features.",
        )

    try:
        expected_features = getattr(model, "feature_names_in_", None)

        if expected_features is not None:
            missing = [
                str(name)
                for name in expected_features
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

            # Preserve the feature names and order used during training.
            input_df = pd.DataFrame(
                [[features[name] for name in expected_features]],
                columns=list(expected_features),
            )
        else:
            input_df = pd.DataFrame([features])

        prediction = model.predict(input_df)[0]

        if isinstance(prediction, np.generic):
            prediction = prediction.item()

        result = {"prediction": prediction}

        # Include class probabilities when supported.
        if callable(getattr(model, "predict_proba", None)):
            probabilities = model.predict_proba(input_df)[0]
            classes = getattr(
                model,
                "classes_",
                range(len(probabilities)),
            )

            result["probabilities"] = {
                str(label): float(probability)
                for label, probability in zip(classes, probabilities)
            }

        # Return saved threshold and labels when available.
        if "threshold" in model_metadata:
            result["threshold"] = float(model_metadata["threshold"])

        if "positive_label" in model_metadata:
            result["positive_label"] = str(
                model_metadata["positive_label"]
            )

        if "negative_label" in model_metadata:
            result["negative_label"] = str(
                model_metadata["negative_label"]
            )

        return result

    except HTTPException:
        raise
    except Exception as exc:
        print(f"PREDICTION ERROR: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )
