# Traffic Accidents Classification - Deployment Server

A web-based prediction server for classifying traffic accident outcomes using 7 machine learning models.

## Overview

This deployment provides:
- **FastAPI Backend**: REST API for predictions and model evaluation
- **Web UI**: Interactive frontend for non-expert users
- **7 ML Models**: Gradient Boosting, Random Forest, Decision Tree, MLP, KNN, Logistic Regression, Naive Bayes

**Target Variable:** `crash_type`
- Class 0: NO INJURY / DRIVE AWAY
- Class 1: INJURY AND / OR TOW DUE TO CRASH

**Best Model:** Gradient Boosting (F1: 81.7%)

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Navigate to the deployment directory:
   ```bash
   cd deployment
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate model artifacts** (required on first setup):
   ```bash
   python train_and_save_models.py
   ```

   This trains all 7 models and saves them to `models/`. The `.joblib` files are not included in the repository due to size constraints.

### Starting the Server

Run the FastAPI server with uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Or with auto-reload for development:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will start and display:
```
INFO:     Loaded 139 feature names
INFO:     Loaded model: naive_bayes from models/naive_bayes.joblib
INFO:     Loaded model: logistic_regression from models/logistic_regression.joblib
INFO:     Loaded model: decision_tree from models/decision_tree.joblib
INFO:     Loaded model: mlp from models/mlp.joblib
INFO:     Loaded model: knn from models/knn.joblib
INFO:     Loaded model: random_forest from models/random_forest.joblib
INFO:     Loaded model: gradient_boosting from models/gradient_boosting.joblib
INFO:     Successfully loaded 7 models
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Accessing the Client

Open your web browser and navigate to:

```
http://localhost:8000
```

## Using the Web Interface

### Single Prediction Tab

1. Adjust the input sliders and dropdowns to describe a crash scenario:
   - **Time Info**: Hour, day of week, month
   - **Crash Details**: Number of vehicles, damage level
   - **Injury Info**: Fatal, incapacitating, no indication counts
   - **Conditions**: Traffic control, weather, lighting, crash type, etc.

2. Select a model from the dropdown (or use the default Gradient Boosting)

3. Click **Predict** to see the result

### Performance Analysis Tab

1. Upload a CSV file with validation data (must include `crash_type` column)
2. Select which models to evaluate
3. Click **Evaluate** to see accuracy, precision, recall, F1, and confusion matrices

### Models Info Tab

View details about all 7 available models including:
- Hyperparameters
- Training metrics
- Ranking by F1 score

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/models` | GET | List available models |
| `/api/predict-single` | POST | Single instance prediction |
| `/api/evaluate-models` | POST | Batch evaluation |
| `/api/features` | GET | List feature names |
| `/api/pipeline` | GET | Pipeline description |
| `/health` | GET | Health check |

### Example API Calls

**Single Prediction:**
```bash
curl -X POST http://localhost:8000/api/predict-single \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "damage": 0.5,
      "injuries_no_indication": 0.5,
      "crash_hour": 0.5,
      "first_crash_type_REAR END": true
    },
    "model_name": "gradient_boosting"
  }'
```

**List Models:**
```bash
curl http://localhost:8000/api/models
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

## Retraining Models

If you need to retrain the models with updated data:

```bash
python train_and_save_models.py
```

This will:
1. Load the prepared data from `../classification/ds1_traffic_accidents/data/prepared/best_model.csv`
2. Train all 7 models with optimized hyperparameters
3. Save model artifacts to `models/`
4. Generate configuration files

## Project Structure

```
deployment/
├── app.py                      # FastAPI server
├── train_and_save_models.py    # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── ARCHITECTURE.md             # System architecture
├── FILE_FORMATS.md             # API schemas and formats
├── models/                     # Model artifacts
│   ├── config.json
│   ├── pipeline_description.json
│   ├── feature_info.json
│   └── *.joblib                # Trained models
├── templates/
│   └── index.html              # Web UI
└── static/
    ├── script.js               # Frontend logic
    └── styles.css              # Styling
```

## Dependencies

- fastapi
- uvicorn
- pandas
- numpy
- scikit-learn
- joblib
- python-multipart

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and data flows
- [FILE_FORMATS.md](FILE_FORMATS.md) - API request/response schemas

## Troubleshooting

### Server won't start

1. Ensure you're in the `deployment/` directory
2. Check that model files exist in `models/`
3. Verify Python dependencies are installed

### Models not loading

Run the training script to regenerate models:
```bash
python train_and_save_models.py
```

### Port already in use

Use a different port:
```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

### CORS issues

The server allows all origins by default. For production, configure CORS in `app.py`.
