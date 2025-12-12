"""
FastAPI Backend for Traffic Accidents Classification Prediction Server

This server provides endpoints for:
- GET /api/models - List available models and metadata
- POST /api/predict-single - Single-instance prediction
- POST /api/evaluate-models - Performance estimation on uploaded validation data
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import io

# Initialize FastAPI app
app = FastAPI(
    title="Traffic Accidents Classification API",
    description="Prediction server for classifying traffic accidents",
    version="1.0.0"
)

# Global variables for models and configuration
models = {}
config = None
feature_names = []

# Paths
MODELS_DIR = "models"
CONFIG_PATH = os.path.join(MODELS_DIR, "config.json")


def load_models():
    """Load all models and configuration at startup."""
    global models, config, feature_names

    # Load configuration
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded successfully")
    except FileNotFoundError:
        raise RuntimeError(f"Configuration file not found: {CONFIG_PATH}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in configuration file: {e}")

    # Load feature names
    feature_names = config.get("features", [])
    print(f"Loaded {len(feature_names)} feature names")

    # Load each model
    for model_key, model_info in config.get("models", {}).items():
        model_file = model_info.get("file")
        model_path = os.path.join(MODELS_DIR, model_file)

        try:
            models[model_key] = joblib.load(model_path)
            print(f"Loaded model: {model_key} from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model file not found: {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load model {model_key}: {e}")

    if not models:
        raise RuntimeError("No models were loaded successfully")

    print(f"Successfully loaded {len(models)} models")


# Load models at startup
@app.on_event("startup")
async def startup_event():
    load_models()


# ============== Pydantic Models ==============

class SinglePredictionRequest(BaseModel):
    """Request model for single instance prediction."""
    features: Dict[str, Any]
    model_name: Optional[str] = None


class SinglePredictionResponse(BaseModel):
    """Response model for single instance prediction."""
    model_used: str
    predicted_class: int
    predicted_label: str
    confidence: Optional[float] = None


class ModelInfo(BaseModel):
    """Model information response."""
    name: str
    key: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    is_default: bool


class ModelsListResponse(BaseModel):
    """Response model for listing all models."""
    project: str
    description: str
    default_model: str
    target: Dict[str, Any]
    models: List[ModelInfo]
    total_features: int


class EvaluationResult(BaseModel):
    """Evaluation result for a single model."""
    model_name: str
    model_key: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]


class EvaluationResponse(BaseModel):
    """Response model for model evaluation."""
    num_samples: int
    results: List[EvaluationResult]


# ============== API Endpoints ==============

@app.get("/api/models", response_model=ModelsListResponse)
async def get_models():
    """
    Get a list of all available models and their metadata.

    Returns information about each model including:
    - Name and identifier
    - Hyperparameters used for training
    - Performance metrics (accuracy, precision, recall, F1)
    - Whether it's the default model
    """
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")

    models_list = []
    default_model = config.get("default_model", "")

    for model_key, model_info in config.get("models", {}).items():
        models_list.append(ModelInfo(
            name=model_info.get("name", model_key),
            key=model_key,
            hyperparameters=model_info.get("hyperparameters", {}),
            metrics=model_info.get("metrics", {}),
            is_default=(model_key == default_model)
        ))

    # Sort by F1 score descending
    models_list.sort(key=lambda x: x.metrics.get("f1", 0), reverse=True)

    return ModelsListResponse(
        project=config.get("project", "Traffic Accidents Classification"),
        description=config.get("description", ""),
        default_model=default_model,
        target=config.get("target", {}),
        models=models_list,
        total_features=len(feature_names)
    )


@app.post("/api/predict-single", response_model=SinglePredictionResponse)
async def predict_single(request: SinglePredictionRequest):
    """
    Make a prediction for a single instance.

    The request should contain:
    - features: Dictionary with feature names as keys and values
    - model_name: (optional) The model to use for prediction. If not specified,
                  the default model (Gradient Boosting) will be used.

    Returns:
    - model_used: Name of the model used
    - predicted_class: Numeric class (0 or 1)
    - predicted_label: Human-readable class label
    - confidence: Prediction probability (if available)
    """
    # Determine which model to use
    model_name = request.model_name or config.get("default_model")

    if model_name not in models:
        available = list(models.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found. Available models: {available}"
        )

    model = models[model_name]
    model_info = config.get("models", {}).get(model_name, {})

    # Prepare feature vector
    try:
        # Get feature types
        numeric_features = config.get("feature_types", {}).get("numeric", [])
        boolean_features = config.get("feature_types", {}).get("boolean", [])

        # Create a dictionary with proper default values by type
        default_values = {}
        for feat in feature_names:
            if feat in boolean_features:
                default_values[feat] = False
            else:
                default_values[feat] = 0.0

        # Create DataFrame with proper dtypes from the start
        feature_vector = pd.DataFrame([default_values])

        # Update with provided values
        for feat_name, feat_value in request.features.items():
            if feat_name in feature_names:
                if feat_name in boolean_features:
                    feature_vector.loc[0, feat_name] = bool(feat_value)
                else:
                    feature_vector.loc[0, feat_name] = float(feat_value)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error preparing feature vector: {str(e)}"
        )

    # Make prediction
    try:
        prediction = model.predict(feature_vector)[0]
        prediction = int(prediction)

        # Get prediction probability if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feature_vector)[0]
            confidence = float(max(proba))

        # Get class label
        target_classes = config.get("target", {}).get("classes", {})
        predicted_label = target_classes.get(str(prediction), f"Class {prediction}")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

    return SinglePredictionResponse(
        model_used=model_info.get("name", model_name),
        predicted_class=prediction,
        predicted_label=predicted_label,
        confidence=confidence
    )


@app.post("/api/evaluate-models", response_model=EvaluationResponse)
async def evaluate_models(
    file: UploadFile = File(...),
    model_names: Optional[str] = Form(None)
):
    """
    Evaluate model(s) on an uploaded validation dataset.

    The uploaded file should be a CSV with the same format as the training data,
    including the target column 'crash_type'.

    Parameters:
    - file: CSV file with validation data
    - model_names: (optional) Comma-separated list of model names to evaluate.
                   If not specified, all models will be evaluated.

    Returns performance metrics for each model:
    - accuracy, precision, recall, F1 score
    - confusion matrix
    """
    # Parse model names
    if model_names:
        selected_models = [m.strip() for m in model_names.split(",")]
        # Validate model names
        invalid_models = [m for m in selected_models if m not in models]
        if invalid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model names: {invalid_models}. Available: {list(models.keys())}"
            )
    else:
        selected_models = list(models.keys())

    # Read uploaded file
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error reading CSV file: {str(e)}"
        )

    # Validate that target column exists
    target_name = config.get("target", {}).get("name", "crash_type")
    if target_name not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_name}' not found in uploaded file. "
                   f"Available columns: {list(df.columns)}"
        )

    # Split features and target
    try:
        y_true = df[target_name].values
        X = df.drop(columns=[target_name])

        # Ensure features match expected format
        # Add missing columns with default values
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0.0 if col in config.get("feature_types", {}).get("numeric", []) else False

        # Reorder columns to match training data
        X = X[feature_names]

        # Convert boolean features
        boolean_features = config.get("feature_types", {}).get("boolean", [])
        for col in boolean_features:
            if col in X.columns:
                X[col] = X[col].astype(bool)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error preparing validation data: {str(e)}"
        )

    # Evaluate each model
    results = []
    for model_name in selected_models:
        model = models[model_name]
        model_info = config.get("models", {}).get(model_name, {})

        try:
            y_pred = model.predict(X)

            # Calculate metrics
            accuracy = float(accuracy_score(y_true, y_pred))
            precision = float(precision_score(y_true, y_pred, zero_division=0))
            recall = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            cm = confusion_matrix(y_true, y_pred).tolist()

            results.append(EvaluationResult(
                model_name=model_info.get("name", model_name),
                model_key=model_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                confusion_matrix=cm
            ))

        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            # Continue with other models

    if not results:
        raise HTTPException(
            status_code=500,
            detail="Failed to evaluate any models"
        )

    # Sort by F1 score
    results.sort(key=lambda x: x.f1, reverse=True)

    return EvaluationResponse(
        num_samples=len(y_true),
        results=results
    )


@app.get("/api/features")
async def get_features():
    """Get the list of features and their types."""
    return {
        "features": feature_names,
        "feature_types": config.get("feature_types", {}),
        "total": len(feature_names)
    }


@app.get("/api/pipeline")
async def get_pipeline():
    """Get the preparation pipeline description."""
    pipeline_path = os.path.join(MODELS_DIR, "pipeline_description.json")
    try:
        with open(pipeline_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Pipeline description not found")


# ============== Static Files and Frontend ==============

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page."""
    return FileResponse("templates/index.html")


# ============== Health Check ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "config_loaded": config is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
