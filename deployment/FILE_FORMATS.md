# File Formats for Prediction Objects

This document describes the JSON schemas and file formats used for prediction requests, responses, and configuration in the Traffic Accidents Classification API.

## API Request/Response Formats

### Single Prediction Request

**Endpoint:** `POST /api/predict-single`

**Content-Type:** `application/json`

```json
{
  "features": {
    "crash_date": 0.5,
    "intersection_related_i": 1,
    "damage": 0.3,
    "num_units": 0.4,
    "injuries_fatal": 0.0,
    "injuries_incapacitating": 0.0,
    "injuries_no_indication": 0.7,
    "crash_hour": 0.5,
    "crash_day_of_week": 0.5,
    "crash_month": 0.5,
    "traffic_control_device_TRAFFIC SIGNAL": true,
    "traffic_control_device_NO CONTROLS": false,
    "weather_condition_CLOUDY/OVERCAST": false,
    "lighting_condition_DAYLIGHT": true,
    "first_crash_type_REAR END": true,
    "trafficway_type_NOT DIVIDED": true,
    "alignment_STRAIGHT AND LEVEL": true,
    "road_defect_NO DEFECTS": true,
    "prim_contributory_cause_UNABLE TO DETERMINE": true
  },
  "model_name": "gradient_boosting"
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `features` | object | Yes | Dictionary of feature names to values |
| `model_name` | string | No | Model identifier (null = use default) |

**Feature Types:**

| Type | Format | Example |
|------|--------|---------|
| Numeric | float (0.0 - 1.0, MinMax scaled) | `"damage": 0.5` |
| Boolean (one-hot) | boolean | `"first_crash_type_REAR END": true` |
| Integer | number | `"intersection_related_i": 1` |

---

### Single Prediction Response

```json
{
  "model_used": "Gradient Boosting",
  "predicted_class": 0,
  "predicted_label": "NO INJURY / DRIVE AWAY",
  "confidence": 0.8523
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `model_used` | string | Human-readable name of the model used |
| `predicted_class` | integer | Predicted class (0 or 1) |
| `predicted_label` | string | Human-readable class label |
| `confidence` | float | Prediction probability (0.0 - 1.0), null if model doesn't support probabilities |

**Class Labels:**

| Class | Label |
|-------|-------|
| 0 | NO INJURY / DRIVE AWAY |
| 1 | INJURY AND / OR TOW DUE TO CRASH |

---

### Models List Response

**Endpoint:** `GET /api/models`

```json
{
  "project": "Traffic Accidents Classification",
  "description": "Prediction server for classifying traffic accidents based on crash type",
  "default_model": "gradient_boosting",
  "target": {
    "name": "crash_type",
    "classes": {
      "0": "NO INJURY / DRIVE AWAY",
      "1": "INJURY AND / OR TOW DUE TO CRASH"
    }
  },
  "models": [
    {
      "name": "Gradient Boosting",
      "key": "gradient_boosting",
      "hyperparameters": {
        "max_depth": 5,
        "learning_rate": 0.1,
        "n_estimators": 750
      },
      "metrics": {
        "accuracy": 0.8237,
        "precision": 0.8499,
        "recall": 0.7862,
        "f1": 0.8168
      },
      "is_default": true
    }
  ],
  "total_features": 139
}
```

---

### Batch Evaluation Request

**Endpoint:** `POST /api/evaluate-models`

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | CSV file with features and `crash_type` target |
| `model_names` | string | No | Comma-separated model keys (default: all models) |

**CSV File Format:**

```csv
crash_date,damage,num_units,injuries_fatal,...,crash_type
0.834,0.321,0.457,0.0,...,0
0.652,0.891,0.457,0.0,...,1
```

---

### Batch Evaluation Response

```json
{
  "num_samples": 1000,
  "results": [
    {
      "model_name": "Gradient Boosting",
      "model_key": "gradient_boosting",
      "accuracy": 0.8240,
      "precision": 0.8501,
      "recall": 0.7865,
      "f1": 0.8171,
      "confusion_matrix": [
        [412, 88],
        [107, 393]
      ]
    },
    {
      "model_name": "Random Forest",
      "model_key": "random_forest",
      "accuracy": 0.7932,
      "precision": 0.8018,
      "recall": 0.7789,
      "f1": 0.7902,
      "confusion_matrix": [
        [398, 102],
        [111, 389]
      ]
    }
  ]
}
```

**Confusion Matrix Format:**

```
                  Predicted
                  0      1
Actual    0    [TN,    FP]
          1    [FN,    TP]
```

---

## Configuration File Formats

### config.json

Main configuration file loaded at server startup.

```json
{
  "project": "Traffic Accidents Classification",
  "description": "Prediction server for classifying traffic accidents based on crash type",
  "target": {
    "name": "crash_type",
    "classes": {
      "0": "NO INJURY / DRIVE AWAY",
      "1": "INJURY AND / OR TOW DUE TO CRASH"
    }
  },
  "features": [
    "crash_date",
    "intersection_related_i",
    "damage",
    "num_units",
    "injuries_fatal",
    "injuries_incapacitating",
    "injuries_no_indication",
    "crash_hour",
    "crash_day_of_week",
    "crash_month",
    "traffic_control_device_BICYCLE CROSSING SIGN",
    "traffic_control_device_DELINEATORS",
    "..."
  ],
  "feature_types": {
    "numeric": [
      "crash_date",
      "intersection_related_i",
      "damage",
      "num_units",
      "injuries_fatal",
      "injuries_incapacitating",
      "injuries_no_indication",
      "crash_hour",
      "crash_day_of_week",
      "crash_month"
    ],
    "boolean": [
      "traffic_control_device_BICYCLE CROSSING SIGN",
      "traffic_control_device_DELINEATORS",
      "..."
    ]
  },
  "default_model": "gradient_boosting",
  "models": {
    "gradient_boosting": {
      "name": "Gradient Boosting",
      "file": "gradient_boosting.joblib",
      "hyperparameters": {
        "max_depth": 5,
        "learning_rate": 0.1,
        "n_estimators": 750
      },
      "metrics": {
        "accuracy": 0.8237,
        "precision": 0.8499,
        "recall": 0.7862,
        "f1": 0.8168
      }
    },
    "naive_bayes": {
      "name": "Naive Bayes (Bernoulli)",
      "file": "naive_bayes.joblib",
      "hyperparameters": {
        "type": "BernoulliNB"
      },
      "metrics": {
        "accuracy": 0.7102,
        "precision": 0.7105,
        "recall": 0.7094,
        "f1": 0.7099
      }
    }
  },
  "pipeline": {
    "file": "pipeline_description.json"
  }
}
```

---

### pipeline_description.json

Describes the data preparation pipeline.

```json
{
  "description": "Data preparation pipeline for traffic accidents classification",
  "steps": [
    {
      "name": "Data loading",
      "description": "Load raw CSV data"
    },
    {
      "name": "Feature extraction",
      "description": "Extract date components (crash_hour, crash_day_of_week, crash_month) from crash_date"
    },
    {
      "name": "Target encoding",
      "description": "Binary encoding of crash_type: 0='NO INJURY / DRIVE AWAY', 1='INJURY AND / OR TOW DUE TO CRASH'"
    },
    {
      "name": "Missing value handling",
      "description": "Drop columns that are entirely empty, drop rows with any missing values"
    },
    {
      "name": "Categorical encoding",
      "description": "One-hot encoding for categorical variables"
    },
    {
      "name": "Numeric scaling",
      "description": "MinMax scaling applied to numeric features to normalize values between 0 and 1"
    },
    {
      "name": "Feature selection",
      "description": "Removed highly correlated features and features with low variance"
    }
  ],
  "input_features": {
    "numeric": ["crash_date", "intersection_related_i", "damage", "num_units",
                "injuries_fatal", "injuries_incapacitating", "injuries_no_indication",
                "crash_hour", "crash_day_of_week", "crash_month"],
    "categorical": ["traffic_control_device", "weather_condition", "lighting_condition",
                   "first_crash_type", "trafficway_type", "alignment",
                   "roadway_surface_cond", "road_defect", "prim_contributory_cause"]
  },
  "target": {
    "name": "crash_type",
    "classes": {
      "0": "NO INJURY / DRIVE AWAY",
      "1": "INJURY AND / OR TOW DUE TO CRASH"
    }
  }
}
```

---

### feature_info.json

Feature metadata for reference.

```json
{
  "feature_names": [
    "crash_date",
    "intersection_related_i",
    "damage",
    "..."
  ],
  "numeric_features": [
    "crash_date",
    "intersection_related_i",
    "damage",
    "num_units",
    "injuries_fatal",
    "injuries_incapacitating",
    "injuries_no_indication",
    "crash_hour",
    "crash_day_of_week",
    "crash_month"
  ],
  "boolean_features": [
    "traffic_control_device_BICYCLE CROSSING SIGN",
    "traffic_control_device_DELINEATORS",
    "..."
  ],
  "total_features": 139
}
```

---

## Model Artifact Format

Models are saved using `joblib` serialization.

**File Extension:** `.joblib`

**Supported Models:**

| File | scikit-learn Class |
|------|-------------------|
| `naive_bayes.joblib` | `BernoulliNB` |
| `logistic_regression.joblib` | `LogisticRegression` |
| `decision_tree.joblib` | `DecisionTreeClassifier` |
| `mlp.joblib` | `MLPClassifier` |
| `knn.joblib` | `KNeighborsClassifier` |
| `random_forest.joblib` | `RandomForestClassifier` |
| `gradient_boosting.joblib` | `GradientBoostingClassifier` |

**Loading Example:**

```python
import joblib

model = joblib.load('models/gradient_boosting.joblib')
predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

---

## Feature Value Ranges

### Numeric Features (MinMax Scaled: 0.0 - 1.0)

| Feature | Description | UI Display |
|---------|-------------|------------|
| `crash_date` | Normalized date | N/A (hidden) |
| `damage` | Damage amount | $500 / $1,500 / $1,500+ |
| `num_units` | Number of vehicles | 1 - 5+ |
| `injuries_fatal` | Fatal injuries count | 0 - 4 |
| `injuries_incapacitating` | Incapacitating injuries | 0 - 4 |
| `injuries_no_indication` | No injury indication count | Low / Med / High |
| `crash_hour` | Hour of day (0-23) | 00:00 - 23:00 |
| `crash_day_of_week` | Day (0=Mon, 6=Sun) | Mon - Sun |
| `crash_month` | Month (0=Jan, 11=Dec) | Jan - Dec |
| `intersection_related_i` | Intersection related | 0 or 1 |

### Boolean Features (One-Hot Encoded)

Only one feature per category should be `true`:

| Category | Example Features |
|----------|-----------------|
| `traffic_control_device_*` | TRAFFIC SIGNAL, STOP SIGN/FLASHER, NO CONTROLS |
| `weather_condition_*` | CLOUDY/OVERCAST, SNOW, FOG/SMOKE/HAZE |
| `lighting_condition_*` | DAYLIGHT, DARKNESS, DAWN, DUSK |
| `first_crash_type_*` | REAR END, ANGLE, HEAD ON, TURNING |
| `trafficway_type_*` | NOT DIVIDED, FOUR WAY, T-INTERSECTION |
| `alignment_*` | STRAIGHT AND LEVEL, CURVE ON GRADE |
| `road_defect_*` | NO DEFECTS, RUT, HOLES |
| `prim_contributory_cause_*` | UNABLE TO DETERMINE, FOLLOWING TOO CLOSELY |
