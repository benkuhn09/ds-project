"""
Script to train and save all classification models and the preparation pipeline.
This script trains models on the prepared traffic accidents data and saves them
using joblib for later deployment.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

# Paths
DATA_PATH = "../classification/ds1_traffic_accidents/data/prepared/best_model.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Load prepared data
print("Loading prepared data...")
df = pd.read_csv(DATA_PATH)

target_name = "crash_type"
target = df[target_name]
features = df.drop(columns=[target_name])

# Get feature names and types
feature_names = features.columns.tolist()
numeric_features = features.select_dtypes(include=['float64', 'int64']).columns.tolist()
boolean_features = features.select_dtypes(include=['bool']).columns.tolist()

print(f"Total features: {len(feature_names)}")
print(f"Numeric features: {len(numeric_features)}")
print(f"Boolean features: {len(boolean_features)}")

# Train/test split
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.3, random_state=42, stratify=target
)

print(f"Training set size: {len(features_train)}")
print(f"Test set size: {len(features_test)}")

# Define models with their best hyperparameters from ds1_lab4_modelling.ipynb
models_config = {
    "naive_bayes": {
        "model": BernoulliNB(),
        "name": "Naive Bayes (Bernoulli)",
        "hyperparameters": {
            "type": "BernoulliNB"
        }
    },
    "logistic_regression": {
        "model": LogisticRegression(max_iter=500, solver='liblinear', penalty='l2', verbose=False),
        "name": "Logistic Regression",
        "hyperparameters": {
            "penalty": "l2",
            "max_iter": 500,
            "solver": "liblinear"
        }
    },
    "decision_tree": {
        "model": DecisionTreeClassifier(max_depth=13, criterion='gini', min_impurity_decrease=0),
        "name": "Decision Tree",
        "hyperparameters": {
            "criterion": "gini",
            "max_depth": 13,
            "min_impurity_decrease": 0
        }
    },
    "mlp": {
        "model": MLPClassifier(
            activation='logistic',
            learning_rate='constant',
            learning_rate_init=0.005,
            max_iter=2500,
            solver='sgd',
            warm_start=True,
            random_state=42
        ),
        "name": "Multi-Layer Perceptron",
        "hyperparameters": {
            "activation": "logistic",
            "learning_rate": "constant",
            "learning_rate_init": 0.005,
            "max_iter": 2500,
            "solver": "sgd"
        }
    },
    "knn": {
        "model": KNeighborsClassifier(n_neighbors=19, metric='manhattan'),
        "name": "K-Nearest Neighbors",
        "hyperparameters": {
            "n_neighbors": 19,
            "metric": "manhattan"
        }
    },
    "random_forest": {
        "model": RandomForestClassifier(max_depth=7, max_features=0.9, n_estimators=100, random_state=42),
        "name": "Random Forest",
        "hyperparameters": {
            "max_depth": 7,
            "max_features": 0.9,
            "n_estimators": 100
        }
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier(max_depth=5, learning_rate=0.1, n_estimators=750, random_state=42),
        "name": "Gradient Boosting",
        "hyperparameters": {
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_estimators": 750
        }
    }
}

# Train and evaluate models
results = {}
print("\nTraining models...")

for model_key, model_info in models_config.items():
    print(f"\n  Training {model_info['name']}...")
    model = model_info["model"]

    # Train
    model.fit(features_train, target_train)

    # Predict
    predictions = model.predict(features_test)

    # Evaluate
    metrics = {
        "accuracy": float(accuracy_score(target_test, predictions)),
        "precision": float(precision_score(target_test, predictions)),
        "recall": float(recall_score(target_test, predictions)),
        "f1": float(f1_score(target_test, predictions))
    }

    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    print(f"    F1 Score: {metrics['f1']:.4f}")

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{model_key}.joblib")
    joblib.dump(model, model_path)
    print(f"    Saved to: {model_path}")

    results[model_key] = {
        "name": model_info["name"],
        "file": f"{model_key}.joblib",
        "hyperparameters": model_info["hyperparameters"],
        "metrics": metrics
    }

# Determine the best model based on F1 score
best_model_key = max(results.keys(), key=lambda k: results[k]["metrics"]["f1"])
print(f"\nBest model: {results[best_model_key]['name']} (F1: {results[best_model_key]['metrics']['f1']:.4f})")

# Create preparation pipeline description
# Since the data is already prepared, we create an identity pipeline
# that expects data in the same format as the training data
pipeline_description = {
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
            "description": "One-hot encoding for categorical variables (traffic_control_device, weather_condition, lighting_condition, first_crash_type, trafficway_type, alignment, roadway_surface_cond, road_defect, prim_contributory_cause)"
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

# Save pipeline description
pipeline_path = os.path.join(MODELS_DIR, "pipeline_description.json")
with open(pipeline_path, 'w') as f:
    json.dump(pipeline_description, f, indent=2)
print(f"\nPipeline description saved to: {pipeline_path}")

# Create the main configuration file
config = {
    "project": "Traffic Accidents Classification",
    "description": "Prediction server for classifying traffic accidents based on crash type",
    "target": {
        "name": "crash_type",
        "classes": {
            "0": "NO INJURY / DRIVE AWAY",
            "1": "INJURY AND / OR TOW DUE TO CRASH"
        }
    },
    "features": feature_names,
    "feature_types": {
        "numeric": numeric_features,
        "boolean": boolean_features
    },
    "default_model": best_model_key,
    "models": results,
    "pipeline": {
        "file": "pipeline_description.json"
    }
}

# Save configuration
config_path = os.path.join(MODELS_DIR, "config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"Configuration saved to: {config_path}")

# Save feature names for reference
feature_info_path = os.path.join(MODELS_DIR, "feature_info.json")
with open(feature_info_path, 'w') as f:
    json.dump({
        "feature_names": feature_names,
        "numeric_features": numeric_features,
        "boolean_features": boolean_features,
        "total_features": len(feature_names)
    }, f, indent=2)
print(f"Feature info saved to: {feature_info_path}")

print("\n" + "="*50)
print("Model training and saving complete!")
print("="*50)
