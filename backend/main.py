from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# --------------------------------
# Create FastAPI app
# --------------------------------
app = FastAPI(title="Diabetes Risk Prediction API")

# --------------------------------
# Enable CORS (frontend access)
# --------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------
# Load trained BALANCED model & scaler
# --------------------------------
model = joblib.load("diabetes_random_forest_balanced.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------------
# Input schema (MUST match dataset)
# --------------------------------
class PatientInput(BaseModel):
    age: int
    gender: int                 # 0 = Male, 1 = Female
    pulse_rate: int
    systolic_bp: int
    diastolic_bp: int
    glucose: float
    height: float
    weight: float
    bmi: float
    family_diabetes: int
    hypertensive: int
    family_hypertension: int
    cardiovascular_disease: int
    stroke: int

# --------------------------------
# Home endpoint
# --------------------------------
@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}

# --------------------------------
# Prediction endpoint (FIXED LOGIC)
# --------------------------------
@app.post("/predict")
def predict_diabetes(data: PatientInput):

    # Convert input to numpy array
    input_data = np.array([[
        data.age,
        data.gender,
        data.pulse_rate,
        data.systolic_bp,
        data.diastolic_bp,
        data.glucose,
        data.height,
        data.weight,
        data.bmi,
        data.family_diabetes,
        data.hypertensive,
        data.family_hypertension,
        data.cardiovascular_disease,
        data.stroke
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Get probability from model
    probability = model.predict_proba(input_scaled)[0][1]

    # ✅ MEDICAL DECISION THRESHOLD (IMPORTANT FIX)
    # Lower threshold to avoid missing high-risk patients
    if probability >= 0.35:
        prediction = 1
    else:
        prediction = 0

    return {
        "diabetes": "Yes" if prediction == 1 else "No",
        "risk_probability": round(float(probability), 2)
    }
