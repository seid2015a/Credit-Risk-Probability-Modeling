# src/train.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from scipy.stats import uniform, randint # For RandomizedSearchCV

# Ensure necessary directories exist
MODELS_DIR = '../models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Set MLflow tracking URI (can be local 'mlruns/' or remote server)
mlflow.set_tracking_uri("http://127.0.0.1:5000") # Local MLflow UI
# Set experiment name
mlflow.set_experiment("Credit_Risk_Modeling")

def train_risk_probability_model(X_processed, y_target, model_name_suffix="default"):
    """
    Trains various models to assign risk probability (Probability of Default - PD).
    Performs hyperparameter tuning and logs results to MLflow.
    """
    print(f"\n--- Training Risk Probability Model ({model_name_suffix}) ---")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_target, test_size=0.25, random_state=42, stratify=y_target
    )

    print(f"Training data shape: {X_train.shape}, Target shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, Target shape: {y_test.shape}")
    print(f"Distribution of target in train set:\n{y_train.value_counts(normalize=True)}")

    models_to_train = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'),
            'params': {
                'C': uniform(loc=0, scale=4), # Inverse of regularization strength
                'penalty': ['l1', 'l2']
            }
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'max_depth': randint(3, 10),
                'min_samples_leaf': randint(5, 50)
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
            'params': {
                'n_estimators': randint(50, 200),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 10)
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': randint(50, 200),
                'learning_rate': uniform(0.01, 0.2),
                'max_depth': randint(3, 8),
            }
        }
    }

    best_model_overall = None
    best_f1_score = -1
    best_model_name = ""

    for model_type, config in models_to_train.items():
        with mlflow.start_run(run_name=f"{model_type}_Tuning_{model_name_suffix}"):
            print(f"\n--- Running Hyperparameter Tuning for {model_type} ---")
            mlflow.log_param("model_type", model_type)

            # Use RandomizedSearchCV for efficiency, adjust n_iter for more thorough search
            search = RandomizedSearchCV(
                estimator=config['model'],
                param_distributions=config['params'],
                n_iter=10, # Number of parameter settings that are sampled. Adjust for more thorough search.
                cv=5, # 5-fold cross-validation
                scoring='f1', # Optimize for F1-score due to imbalance
                random_state=42,
                n_jobs=-1, # Use all available cores
                verbose=1
            )
            search.fit(X_train, y_train)

            print(f"Best parameters for {model_type}: {search.best_params_}")
            mlflow.log_params(search.best_params_)

            best_estimator = search.best_estimator_

            # Evaluate best estimator on test set
            y_pred = best_estimator.predict(X_test)
            y_proba = best_estimator.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            }
            mlflow.log_metrics(metrics)

            print(f"\nEvaluation Results for {model_type} (Best Estimator):")
            for metric_name, value in metrics.items():
                print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # Log classification report as artifact
            report_str = classification_report(y_test, y_pred, output_dict=False)
            with open("classification_report.txt", "w") as f:
                f.write(report_str)
            mlflow.log_artifact("classification_report.txt")

            # Log the model itself
            mlflow.sklearn.log_model(best_estimator, f"model_{model_type}")

            if f1 > best_f1_score:
                best_f1_score = f1
                best_model_overall = best_estimator
                best_model_name = model_type

    print(f"\nOverall Best Model (by F1-score): {best_model_name} with F1-score: {best_f1_score:.4f}")

    # Register the best model in MLflow Model Registry
    if best_model_overall:
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model_{best_model_name}"
        mlflow.register_model(model_uri=model_uri, name="CreditRiskProbabilityModel")
        print(f"Best model '{best_model_name}' registered in MLflow Model Registry.")

        # Save the best model locally as well for direct loading in predict.py (fallback)
        model_filename = os.path.join(MODELS_DIR, f'risk_probability_model_BEST.joblib')
        joblib.dump(best_model_overall, model_filename)
        print(f"Best risk probability model also saved locally to {model_filename}")

    return best_model_overall, y_proba, y_test # Return last y_proba/y_test for score model

def train_credit_score_model(risk_probabilities, y_target, model_name="credit_score_scaling"):
    """
    Develops a simple model to assign a credit score from risk probability estimates.
    Higher score typically means lower risk (lower PD).
    """
    print(f"\n--- Training Credit Score Model ---")

    min_pd, max_pd = np.min(risk_probabilities), np.max(risk_probabilities)
    min_score, max_score = 300, 850 # FICO-like range

    # Scale probabilities to desired score range
    # Ensure no division by zero if min_pd == max_pd (e.g., all same proba)
    if (max_pd - min_pd) == 0:
        normalized_pd = np.zeros_like(risk_probabilities) # If all same, map to 0.5 normalized
    else:
        normalized_pd = (risk_probabilities - min_pd) / (max_pd - min_pd)
    
    # Invert and scale such that higher PD -> lower score
    credit_scores = min_score + (1 - normalized_pd) * (max_score - min_score)
    credit_scores = np.round(credit_scores).astype(int)

    score_mapping_params = {
        'min_pd_seen': min_pd,
        'max_pd_seen': max_pd,
        'min_score': min_score,
        'max_score': max_score
    }

    score_model_filename = os.path.join(MODELS_DIR, f'{model_name}.joblib')
    joblib.dump(score_mapping_params, score_model_filename)
    print(f"Credit score mapping parameters saved to {score_model_filename}")

    with mlflow.start_run(run_name="Credit_Score_Mapping"):
        mlflow.log_params(score_mapping_params)
        mlflow.log_artifact(score_model_filename)

    # For visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=risk_probabilities, y=credit_scores, hue=y_target, alpha=0.6, palette={0: 'green', 1: 'red'})
    plt.xlabel("Risk Probability (PD)")
    plt.ylabel("Credit Score")
    plt.title("Credit Score vs. Risk Probability (on Test Set)")
    plt.grid(True)
    plt.savefig(os.path.join(MODELS_DIR, "credit_score_vs_pd.png"))
    plt.show()

    return score_mapping_params

def train_optimal_loan_model(X_processed, y_target, model_name="optimal_loan_rules"):
    """
    Develops a rule-based model that predicts the optimal amount and duration of the loan.
    This is based on risk tiers derived from predicted probabilities.
    """
    print(f"\n--- Developing Optimal Loan Recommendation Logic ---")

    # Define risk tiers based on hypothetical PD ranges
    # These thresholds are illustrative and need to be determined by business strategy and risk appetite.
    # The PD thresholds should align with the risk probability model's output.
    loan_rules = {
        'Very Low Risk': {'max_pd': 0.05, 'recommended_amount_range': (8000, 20000), 'recommended_duration_months': (36, 60)},
        'Low Risk': {'max_pd': 0.15, 'recommended_amount_range': (3000, 8000), 'recommended_duration_months': (12, 36)},
        'Medium Risk': {'max_pd': 0.30, 'recommended_amount_range': (1000, 3000), 'recommended_duration_months': (6, 18)},
        'High Risk': {'max_pd': 0.50, 'recommended_amount_range': (200, 1000), 'recommended_duration_months': (3, 12)},
        'Very High Risk': {'max_pd': 1.0, 'recommended_amount_range': (0, 0), 'recommended_duration_months': (0, 0)} # No loan
    }

    loan_rules_filename = os.path.join(MODELS_DIR, f'{model_name}.joblib')
    joblib.dump(loan_rules, loan_rules_filename)
    print(f"Optimal loan recommendation rules saved to {loan_rules_filename}")

    with mlflow.start_run(run_name="Optimal_Loan_Rules"):
        mlflow.log_params({"loan_rule_tiers": str(loan_rules)}) # Log as string
        mlflow.log_artifact(loan_rules_filename)

    return loan_rules


if __name__ == '__main__':
    # Add src to path to import data_processing
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.data_processing import preprocess_data

    # Load processed data from data_processing.py output
    processed_data_path = '../data/processed/processed_transactions.csv'
    pipeline_path = os.path.join(MODELS_DIR, 'preprocessing_pipeline.joblib')

    # Ensure processed data exists by running data_processing.py if needed
    if not os.path.exists(processed_data_path) or not os.path.exists(pipeline_path):
        print(f"Processed data or pipeline not found. Running src/data_processing.py first.")
        # Load dummy raw data (or your actual raw data) for preprocessing
        raw_data_for_prep_path = '../data/raw/transactions.csv'
        if not os.path.exists(raw_data_for_prep_path):
            print("Raw data not found. Please ensure it's in data/raw/ or run EDA notebook to generate dummy data.")
            exit() # Exit if raw data isn't there for fallback processing

        raw_df_for_prep = pd.read_csv(raw_data_for_prep_path)
        processed_df_train, trained_pipeline_components = preprocess_data(raw_df_for_prep, pipeline_path, mode='train')
        processed_df_train.to_csv(processed_data_path, index=False)
        print("Preprocessing completed and processed data saved.")
    
    # Load processed data
    df_processed = pd.read_csv(processed_data_path)
    
    # Separate features (X) and target (y)
    if 'is_high_risk' not in df_processed.columns:
        raise ValueError("Processed data does not contain 'is_high_risk' column. Ensure HighRiskProxyGenerator is correctly applied and data_processing.py saves it.")

    y = df_processed['is_high_risk']
    X = df_processed.drop(columns=['is_high_risk'])

    # Ensure X only contains numerical features after WOE and scaling.
    # The `data_processing.py` should output a DataFrame where these are already processed.
    # If not, you might get errors in model training (e.g., non-numeric data).

    # Train Risk Probability Models with MLflow tracking and hyperparameter tuning
    best_risk_model, best_model_y_proba, best_model_y_test = train_risk_probability_model(X, y, model_name_suffix="FinalRun")

    # Train Credit Score Model using probabilities from the best risk model
    if best_risk_model is not None and best_model_y_proba is not None:
        train_credit_score_model(best_model_y_proba, best_model_y_test, model_name="credit_score_scaling_from_best_model")
    else:
        print("Best risk model not found or probabilities not available. Skipping credit score model training.")

    # Train Optimal Loan Model (rule definition)
    train_optimal_loan_model(X, y)

    print("\nTraining process completed. Check MLflow UI for experiment tracking: `mlflow ui`")
    print("Models and parameters are saved in the 'models' directory.")

