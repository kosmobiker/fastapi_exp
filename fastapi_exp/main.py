from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Model Inference API",
    description="API for making predictions using our trained ML model",
    version="0.1.0"
)

# Define input data model
class PredictionInput(BaseModel):
    features: list[float]

# Define output data model
class PredictionOutput(BaseModel):
    prediction: float
    confidence: float

@app.on_event("startup")
async def load_model():
    """Load the ML model on startup."""
    global model
    try:
        model_path = Path("models") / "model.joblib"
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError("Could not load ML model")

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "ML Model Inference API",
        "status": "active",
        "version": "0.1.0"
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Make a prediction using the loaded model."""
    try:
        # Convert input features to numpy array
        features = np.array(input_data.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probability/confidence if available
        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            confidence = float(np.max(model.predict_proba(features)[0]))
        
        return PredictionOutput(
            prediction=float(prediction),
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error making prediction"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the service is healthy."""
    return {"status": "healthy"}
