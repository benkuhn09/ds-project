# Architecture Documentation

## System Overview

The Traffic Accidents Classification Deployment Server is a web-based prediction system built with FastAPI (backend) and vanilla HTML/CSS/JavaScript (frontend). It serves 7 machine learning models trained on traffic accident data to classify crash outcomes.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT (Browser)                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     templates/index.html                             │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │   Single     │  │  Performance │  │   Models     │               │    │
│  │  │  Prediction  │  │   Analysis   │  │    Info      │               │    │
│  │  │     Tab      │  │     Tab      │  │     Tab      │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│  ┌─────────────────────┐     │     ┌─────────────────────┐                  │
│  │  static/script.js   │◄────┴────►│  static/styles.css  │                  │
│  │  - API calls        │           │  - UI styling       │                  │
│  │  - Form handling    │           │  - Responsive grid  │                  │
│  │  - Slider controls  │           │  - Animations       │                  │
│  └─────────────────────┘           └─────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ HTTP/REST
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SERVER (FastAPI)                                   │
│                              app.py                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         API Endpoints                                │    │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐    │    │
│  │  │ GET /api/models │ │POST /api/predict│ │POST /api/evaluate   │    │    │
│  │  │                 │ │    -single      │ │    -models          │    │    │
│  │  │ List models &   │ │                 │ │                     │    │    │
│  │  │ metadata        │ │ Single instance │ │ Batch evaluation    │    │    │
│  │  │                 │ │ prediction      │ │ on CSV upload       │    │    │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────────┘    │    │
│  │                                                                      │    │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐    │    │
│  │  │ GET /api/       │ │ GET /api/       │ │ GET /health         │    │    │
│  │  │    features     │ │    pipeline     │ │                     │    │    │
│  │  │                 │ │                 │ │ Health check        │    │    │
│  │  │ Feature list    │ │ Pipeline info   │ │                     │    │    │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Model Loading Module                            │    │
│  │  - load_models(): Loads all .joblib files at startup                │    │
│  │  - Reads config.json for model metadata                             │    │
│  │  - Caches models in memory for fast inference                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ File I/O
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MODEL ARTIFACTS                                     │
│                            models/                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     Configuration Files                              │    │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐    │    │
│  │  │   config.json   │ │ pipeline_       │ │  feature_info.json  │    │    │
│  │  │                 │ │ description.json│ │                     │    │    │
│  │  │ - Model list    │ │                 │ │ - Feature names     │    │    │
│  │  │ - Hyperparams   │ │ - Prep steps    │ │ - Numeric/Boolean   │    │    │
│  │  │ - Metrics       │ │ - Input schema  │ │   classification    │    │    │
│  │  │ - Default model │ │ - Target info   │ │                     │    │    │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     Trained Models (.joblib)                         │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐        │    │
│  │  │gradient_   │ │random_     │ │decision_   │ │   mlp      │        │    │
│  │  │boosting    │ │forest      │ │tree        │ │            │        │    │
│  │  │ (Default)  │ │            │ │            │ │            │        │    │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘        │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐                       │    │
│  │  │logistic_   │ │   knn      │ │naive_bayes │                       │    │
│  │  │regression  │ │            │ │            │                       │    │
│  │  └────────────┘ └────────────┘ └────────────┘                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Training data source
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FOLDERS                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ../classification/ds1_traffic_accidents/data/prepared/              │    │
│  │  └── best_model.csv   (234,752 samples, 140 features)               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  train_and_save_models.py                                            │    │
│  │  - Loads prepared data                                               │    │
│  │  - Trains 7 models with best hyperparameters                        │    │
│  │  - Saves models and configuration to models/                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Single Prediction Flow

```
User Input (UI Form)
        │
        ▼
┌─────────────────┐
│  script.js      │  Collects form data, converts to JSON
│  makePrediction │
└────────┬────────┘
         │
         ▼
    POST /api/predict-single
    {
      "features": {...},
      "model_name": "gradient_boosting" | null
    }
         │
         ▼
┌─────────────────┐
│    app.py       │  1. Validate request
│ predict_single  │  2. Build feature vector (139 features)
│                 │  3. Load selected model (or default)
│                 │  4. model.predict() + predict_proba()
└────────┬────────┘
         │
         ▼
    Response JSON
    {
      "model_used": "Gradient Boosting",
      "predicted_class": 0 | 1,
      "predicted_label": "NO INJURY / DRIVE AWAY",
      "confidence": 0.92
    }
         │
         ▼
┌─────────────────┐
│  script.js      │  Display result with styling
│ displayResult   │
└─────────────────┘
```

### 2. Batch Evaluation Flow

```
User uploads CSV file
        │
        ▼
┌─────────────────┐
│  script.js      │  FormData with file + selected models
│ evaluateModels  │
└────────┬────────┘
         │
         ▼
    POST /api/evaluate-models
    multipart/form-data:
      - file: validation.csv
      - model_names: "gradient_boosting,random_forest,..."
         │
         ▼
┌─────────────────┐
│    app.py       │  1. Parse CSV, extract features/target
│ evaluate_models │  2. For each selected model:
│                 │     - predictions = model.predict(X)
│                 │     - Calculate metrics (acc, prec, rec, f1)
│                 │     - Build confusion matrix
│                 │  3. Sort results by F1 score
└────────┬────────┘
         │
         ▼
    Response JSON
    {
      "num_samples": 1000,
      "results": [
        {
          "model_name": "Gradient Boosting",
          "accuracy": 0.82,
          "confusion_matrix": [[400,100],[80,420]]
        },
        ...
      ]
    }
```

### 3. Model Loading Flow (Startup)

```
Server Start (uvicorn)
        │
        ▼
┌─────────────────┐
│ @app.on_event   │
│   ("startup")   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  load_models()  │
│                 │
│  1. Read models/config.json
│  2. Extract feature_names (139 features)
│  3. For each model in config:
│     - Load .joblib file
│     - Store in memory dict
│  4. Validate at least 1 model loaded
└─────────────────┘
         │
         ▼
    Models cached in global `models` dict
    Ready to serve predictions
```

## Module Descriptions

### Server Modules (app.py)

| Module | Description |
|--------|-------------|
| `load_models()` | Startup function that loads config and all model files |
| `get_models()` | Returns list of available models with metrics |
| `predict_single()` | Handles single-instance predictions |
| `evaluate_models()` | Batch evaluation on uploaded CSV data |
| `get_features()` | Returns feature names for frontend |
| `get_pipeline()` | Returns data preparation pipeline description |

### Client Modules (script.js)

| Module | Description |
|--------|-------------|
| `loadModels()` | Fetches model list from API on page load |
| `loadFeatures()` | Fetches feature names for one-hot encoding |
| `makePrediction()` | Collects form data and calls predict endpoint |
| `evaluateModels()` | Uploads CSV and displays evaluation results |
| `updateSliderDisplay()` | Converts slider values to human-readable labels |
| `resetForm()` | Resets all form inputs to defaults |

## Directory Structure

```
deployment/
├── app.py                      # FastAPI server (main entry point)
├── train_and_save_models.py    # Script to train and save models
├── requirements.txt            # Python dependencies
├── models/                     # Model artifacts directory
│   ├── config.json             # Main configuration
│   ├── pipeline_description.json
│   ├── feature_info.json
│   ├── gradient_boosting.joblib
│   ├── random_forest.joblib
│   ├── decision_tree.joblib
│   ├── mlp.joblib
│   ├── logistic_regression.joblib
│   ├── knn.joblib
│   └── naive_bayes.joblib
├── templates/
│   └── index.html              # Main UI template
└── static/
    ├── script.js               # Frontend JavaScript
    └── styles.css              # CSS styling
```
