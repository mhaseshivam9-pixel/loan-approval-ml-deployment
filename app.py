from pathlib import Path
import json
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------
# 1. Application setup
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Loan Approval Prediction API",
    description="Loan approval predictions using a trained ML pipeline",
    version="1.0.0",
)

# --------------------------------------------------
# 2. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "loan_approval_deployment_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
SCHEMA_PATH = MODEL_DIR / "input_schema.json"

model = None
metadata = {}
input_schema = {}
model_load_error = None

# --------------------------------------------------
# 3. Load model, metadata, and input schema
# --------------------------------------------------

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    loaded_object = joblib.load(MODEL_PATH)

    # Support either a model object or a dictionary containing a model.
    if isinstance(loaded_object, dict) and "model" in loaded_object:
        model = loaded_object["model"]
    else:
        model = loaded_object

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
            input_schema = json.load(file)

    logger.info("Model and metadata loaded successfully.")

except Exception as exc:
    model = None
    model_load_error = str(exc)
    logger.exception("Model loading failed.")

# --------------------------------------------------
# 4. Applicant input schema
# Names match your input_schema.json
# --------------------------------------------------

class Applicant(BaseModel):
    no_of_dependents: int = Field(ge=0)
    income_annum: float = Field(ge=0)
    loan_amount: float = Field(ge=0)
    loan_term: int = Field(gt=0)
    cibil_score: int = Field(ge=0)
    total_assets_value: float = Field(ge=0)
    loan_to_income_ratio: float = Field(ge=0)
    asset_to_income_ratio: float = Field(ge=0)
    loan_to_total_assets_ratio: float = Field(ge=0)
    education: str
    self_employed: str

# --------------------------------------------------
# 5. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Loan Approval Prediction API is running",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict",
    }

# --------------------------------------------------
# 6. Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model failed to load. Check Vercel runtime logs.",
        )

    return {
        "status": "ok",
        "model_loaded": True,
        "model_type": metadata.get("model_type", "unknown"),
    }

# --------------------------------------------------
# 7. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(applicant: Applicant):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is unavailable. Check Vercel runtime logs.",
        )

    try:
        # Build a DataFrame using the exact feature names expected
        # by your trained pipeline.
        input_data = pd.DataFrame(
            [applicant.model_dump()]
        )

        # Respect the feature order recorded in input_schema.json.
        expected_columns = input_schema.get("columns")

        if expected_columns:
            input_data = input_data[expected_columns]

        # Generate the model's predicted class.
        prediction = model.predict(input_data)[0]
        predicted_label = str(prediction)

        threshold = float(metadata.get("threshold", 0.5))
        positive_label = str(metadata.get("positive_label", "1"))
        negative_label = str(metadata.get("negative_label", "0"))

        approval_probability = None
        decision = (
            "Approved"
            if predicted_label == positive_label
            else "Rejected"
        )

        # Calculate probability if the pipeline supports it.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(getattr(model, "classes_", []))

            positive_index = next(
                (
                    i for i, label in enumerate(classes)
                    if str(label) == positive_label
                ),
                None,
            )

            if positive_index is not None:
                approval_probability = float(
                    probabilities[positive_index]
                )

                decision = (
                    "Approved"
                    if approval_probability >= threshold
                    else "Rejected"
                )

        return {
            "prediction": predicted_label,
            "decision": decision,
            "approval_probability": approval_probability,
            "threshold": threshold,
        }

    except Exception:
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. Check the input features, "
                "model compatibility, and Vercel runtime logs."
            ),
        )