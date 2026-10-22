from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "loan_approval_deployment_model.joblib"

app = FastAPI(
    title="Loan Approval Prediction API",
    description="User-friendly loan approval prediction application",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_metadata = {}
model_error = None


# --------------------------------------------------
# 2. Load the saved model
# --------------------------------------------------

def load_model():
    global model, model_metadata, model_error

    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        saved_object = joblib.load(MODEL_PATH)

        # The notebook saves a dictionary containing the model.
        if isinstance(saved_object, dict):
            loaded_model = saved_object.get("model")

            if loaded_model is None:
                raise ValueError(
                    "The saved deployment package has no 'model' entry."
                )

            model_metadata = {
                key: value
                for key, value in saved_object.items()
                if key != "model"
            }
        else:
            loaded_model = saved_object
            model_metadata = {}

        if not callable(getattr(loaded_model, "predict", None)):
            raise ValueError(
                "The loaded estimator does not have a predict() method."
            )

        model = loaded_model
        model_error = None

        print("Model loaded successfully.")
        print("Model type:", type(model).__name__)
        print("Metadata keys:", list(model_metadata.keys()))

    except Exception as exc:
        model = None
        model_metadata = {}
        model_error = str(exc)
        print("MODEL LOADING ERROR:", exc)


load_model()


# --------------------------------------------------
# 3. User interface
# --------------------------------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#10243a">

<title>LoanWise | Loan Approval Predictor</title>

<style>
:root {
    --navy: #10243a;
    --blue: #2457d6;
    --teal: #0e9f91;
    --text: #172b45;
    --muted: #64748b;
    --line: #e2e8f0;
    --background: #f3f6fb;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
    background: var(--background);
    color: var(--text);
}

header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 15px;
    padding: 18px 6%;
    background: white;
    border-bottom: 1px solid var(--line);
}

.brand {
    font-size: 22px;
    font-weight: 850;
    color: var(--navy);
    letter-spacing: -0.7px;
}

.brand span {
    color: var(--blue);
}

header a {
    color: var(--muted);
    font-size: 14px;
    text-decoration: none;
}

header a:hover {
    color: var(--blue);
}

.hero {
    background: linear-gradient(120deg, #10243a, #234d83);
    color: white;
    padding: 48px 6% 60px;
}

.hero-inner {
    max-width: 1000px;
    margin: auto;
}

.tag {
    color: #a8f0e5;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.8px;
}

h1 {
    font-size: clamp(30px, 5vw, 45px);
    letter-spacing: -1.2px;
    margin: 14px 0;
}

.hero p {
    color: #d6e2f1;
    line-height: 1.8;
    max-width: 660px;
    margin-bottom: 0;
}

main {
    width: min(960px, 92%);
    margin: -28px auto 42px;
    position: relative;
}

.card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: clamp(20px, 4vw, 36px);
    box-shadow: 0 14px 45px #10243a0d;
}

.section-title {
    margin: 0 0 8px;
    font-size: 22px;
}

.hint {
    margin: 0 0 28px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
}

.grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 21px;
}

label {
    display: block;
    font-size: 13px;
    font-weight: 750;
    margin-bottom: 9px;
}

input, select {
    width: 100%;
    min-height: 48px;
    padding: 12px;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    background: white;
    color: var(--text);
    font: inherit;
}

input:focus, select:focus {
    outline: 3px solid #2457d622;
    border-color: var(--blue);
}

input::placeholder {
    color: #94a3b8;
}

.note {
    margin: 24px 0;
    padding: 16px;
    background: #f0f7ff;
    border: 1px solid #dbeafe;
    border-radius: 12px;
    font-size: 13px;
    color: #34516f;
    line-height: 1.8;
}

button {
    width: 100%;
    min-height: 52px;
    padding: 14px 20px;
    border: none;
    border-radius: 10px;
    background: var(--blue);
    color: white;
    font: inherit;
    font-weight: 800;
    cursor: pointer;
    transition: background .15s ease;
}

button:hover {
    background: #1947b7;
}

button:disabled {
    opacity: .65;
    cursor: wait;
}

.result {
    display: none;
    margin-top: 24px;
    padding: 24px;
    border: 1px solid var(--line);
    border-radius: 14px;
    background: #f8fafc;
}

.result h2 {
    font-size: 23px;
    margin: 0 0 10px;
}

.result p {
    line-height: 1.7;
}

.result-status {
    display: inline-block;
    margin-bottom: 12px;
    padding: 6px 11px;
    border-radius: 30px;
    background: #e0f2fe;
    color: #075985;
    font-size: 12px;
    font-weight: 800;
}

.metric {
    margin: 14px 0;
    color: var(--blue);
    font-size: 34px;
    font-weight: 850;
}

.small {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.8;
}

.error-text {
    color: #b42318;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

footer {
    padding: 0 6% 32px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.8;
    text-align: center;
}

@media (max-width: 600px) {
    header {
        padding: 15px 5%;
    }

    .brand {
        font-size: 19px;
    }

    header a {
        font-size: 12px;
    }

    .hero {
        padding: 36px 5% 52px;
    }

    .grid {
        grid-template-columns: 1fr;
        gap: 18px;
    }
}
</style>
</head>

<body>

<header>
    <div class="brand">Loan<span>Wise</span></div>
    <a href="/docs">API Documentation ↗</a>
</header>

<section class="hero">
    <div class="hero-inner">
        <div class="tag">SMARTER FINANCIAL INSIGHTS</div>
        <h1>Loan Approval Predictor</h1>
        <p>
            Enter the applicant's financial details to receive a prediction
            from our machine-learning model. The form calculates financial
            ratios automatically.
        </p>
    </div>
</section>

<main>
<section class="card">
    <h2 class="section-title">Applicant information</h2>
    <p class="hint">
        Complete all fields. Use the same currency, units, and category
        values as those used when the model was trained.
    </p>

    <form id="loanForm">
        <div class="grid">

            <div>
                <label for="dependents">Number of dependents</label>
                <input id="dependents" type="number" min="0"
                       step="1" value="2" required>
            </div>

            <div>
                <label for="cibil">CIBIL credit score</label>
                <input id="cibil" type="number" min="300" max="900"
                       step="1" value="750" required>
            </div>

            <div>
                <label for="income">Annual income</label>
                <input id="income" type="number" min="0.01"
                       step="any" placeholder="e.g. 600000" required>
            </div>

            <div>
                <label for="loan">Requested loan amount</label>
                <input id="loan" type="number" min="0.01"
                       step="any" placeholder="e.g. 1500000" required>
            </div>

            <div>
                <label for="term">Loan term (training-data units)</label>
                <input id="term" type="number" min="0.01"
                       step="any" value="10" required>
            </div>

            <div>
                <label for="assets">Total asset value</label>
                <input id="assets" type="number" min="0.01"
                       step="any" placeholder="e.g. 5000000" required>
            </div>

            <div>
                <label for="education">Education</label>
                <select id="education" required>
                    <option value="Graduate">Graduate</option>
                    <option value="Not Graduate">Not Graduate</option>
                </select>
            </div>

            <div>
                <label for="selfEmployed">Self-employed?</label>
                <select id="selfEmployed" required>
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>

        </div>

        <div class="note">
            <strong>Automatic financial calculations</strong><br>
            Loan-to-income ratio = loan amount ÷ annual income<br>
            Asset-to-income ratio = total assets ÷ annual income<br>
            Loan-to-assets ratio = loan amount ÷ total assets
        </div>

        <button id="submitBtn" type="submit">
            Check loan prediction
        </button>
    </form>

    <section id="result" class="result" aria-live="polite">
        <div id="resultStatus" class="result-status">
            MODEL RESULT
        </div>
        <h2 id="resultTitle">Prediction result</h2>
        <p id="resultText"></p>
        <div id="probability" class="metric"></div>
        <p id="resultDetails" class="small"></p>
        <p id="errorText" class="small error-text"></p>
        <p class="small">
            This is an automated model prediction, not a loan offer,
            financial advice, or a guarantee of approval. A lender makes
            the actual lending decision.
        </p>
    </section>
</section>
</main>

<footer>
    LoanWise · Machine-learning demonstration<br>
    Avoid entering names, account numbers, or other unnecessary personal
    identifiers. Use applicant information only when authorized.
</footer>

<script>
const form = document.getElementById("loanForm");
const button = document.getElementById("submitBtn");
const result = document.getElementById("result");

function showError(message) {
    result.style.display = "block";
    document.getElementById("resultStatus").textContent = "REQUEST ERROR";
    document.getElementById("resultTitle").textContent =
        "Unable to complete prediction";
    document.getElementById("resultText").textContent =
        "Check the entered values and try again.";
    document.getElementById("probability").textContent = "";
    document.getElementById("resultDetails").textContent = "";
    document.getElementById("errorText").textContent = message;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    result.style.display = "none";
    document.getElementById("errorText").textContent = "";

    const income = Number(document.getElementById("income").value);
    const loan = Number(document.getElementById("loan").value);
    const assets = Number(document.getElementById("assets").value);
    const dependents = Number(document.getElementById("dependents").value);
    const cibil = Number(document.getElementById("cibil").value);
    const term = Number(document.getElementById("term").value);

    if (
        !Number.isFinite(income) || income <= 0 ||
        !Number.isFinite(loan) || loan <= 0 ||
        !Number.isFinite(assets) || assets <= 0 ||
        !Number.isFinite(dependents) || dependents < 0 ||
        !Number.isFinite(cibil) || cibil < 300 || cibil > 900 ||
        !Number.isFinite(term) || term <= 0
    ) {
        showError(
            "Enter valid positive financial values, a non-negative number " +
            "of dependents, and a credit score from 300 to 900."
        );
        return;
    }

    const payload = {
        no_of_dependents: dependents,
        income_annum: income,
        loan_amount: loan,
        loan_term: term,
        cibil_score: cibil,
        total_assets_value: assets,
        loan_to_income_ratio: loan / income,
        asset_to_income_ratio: assets / income,
        loan_to_total_assets_ratio: loan / assets,
        education: document.getElementById("education").value,
        self_employed: document.getElementById("selfEmployed").value
    };

    button.disabled = true;
    button.textContent = "Analyzing details...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            const detail = data.detail;
            throw new Error(
                typeof detail === "string"
                    ? detail
                    : JSON.stringify(detail || data)
            );
        }

        const prediction = String(data.prediction);
        const positiveLabel = String(data.positive_label ?? "1");
        const negativeLabel = String(data.negative_label ?? "0");
        const probabilities = data.probabilities || {};

        const positiveProbability = probabilities[positiveLabel];
        const negativeProbability = probabilities[negativeLabel];

        // Label the prediction using the labels saved with the model.
        // Confirm in the training notebook that the positive label means
        // "approved" before using this interface for public decisions.
        const predictedPositive = prediction === positiveLabel;

        result.style.display = "block";
        document.getElementById("resultStatus").textContent =
            "PREDICTION COMPLETED";

        document.getElementById("resultTitle").textContent =
            predictedPositive
                ? "Predicted: Loan approved"
                : "Predicted: Loan not approved";

        document.getElementById("resultText").textContent =
            "The model returned class " + prediction + ".";

        document.getElementById("probability").textContent =
            positiveProbability === undefined
                ? "Prediction complete"
                : (Number(positiveProbability) * 100).toFixed(2) + "%";

        let details = "Probability of the saved positive class.";

        if (negativeProbability !== undefined) {
            details += " Negative-class probability: " +
                (Number(negativeProbability) * 100).toFixed(2) + "%.";
        }

        if (data.threshold !== undefined) {
            details += " Saved decision threshold: " +
                Number(data.threshold).toFixed(2) + ".";
        }

        document.getElementById("resultDetails").textContent = details;
        document.getElementById("errorText").textContent = "";

        result.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });

    } catch (error) {
        showError(error.message || "An unexpected error occurred.");
    } finally {
        button.disabled = false;
        button.textContent = "Check loan prediction";
    }
});
</script>

</body>
</html>
"""


# --------------------------------------------------
# 4. Homepage
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


# --------------------------------------------------
# 5. Health check
# --------------------------------------------------

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


# --------------------------------------------------
# 6. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(payload: dict):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is unavailable: {model_error}",
        )

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
