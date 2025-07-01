# src/api/main.py

from fastapi import FastAPI, HTTPException
import pandas as pd
import sys
import os
import mlflow.pyfunc # For loading MLflow models
import mlflow.sklearn # Ensure sklearn models are handled if logged directly

# Add parent directory to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import Pydantic models
from src.api.pydantic_models import TransactionData, PredictionResponse

# Import utility functions from predict.py
from src.predict import calculate_credit_score, recommend_loan_terms

app = FastAPI(title="Credit Risk Prediction API")

# --- MLflow Configuration ---
# Set MLflow tracking URI (must match where models were logged during training)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# --- Global Variables for Loaded Models and Pipeline ---
PREPROCESSING_PIPELINE_COMPONENTS = None
RISK_MODEL = None
SCORE_PARAMS = None
LOAN_RULES = None

# --- MLflow Model Registry Name ---
MLFLOW_MODEL_NAME = "CreditRiskProbabilityModel"
# Define the stage you want to load (e.g., "Production", "Staging", or "Latest")
MLFLOW_MODEL_STAGE = "Latest"

@app.on_event("startup")
async def startup_event():
    """Load models and preprocessing pipeline components when the FastAPI application starts up."""
    global PREPROCESSING_PIPELINE_COMPONENTS, RISK_MODEL, SCORE_PARAMS, LOAN_RULES

    try:
        print(f"API: Attempting to load best model '{MLFLOW_MODEL_NAME}' from MLflow Registry (stage: {MLFLOW_MODEL_STAGE})...")
        # Load the latest version of the model from MLflow Model Registry
        RISK_MODEL = mlflow.pyfunc.load_model(model_uri=f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}")
        print("API: Risk probability model loaded successfully from MLflow Registry.")

        # Load other necessary components (preprocessing pipeline, score params, loan rules) locally
        # These were saved using joblib in src/train.py and src/data_processing.py
        models_dir = os.path.join(os.path.dirname(__file__), '../../models')
        preprocessing_pipeline_path = os.path.join(models_dir, 'preprocessing_pipeline.joblib')
        score_params_path = os.path.join(models_dir, 'credit_score_scaling_from_best_model.joblib')
        loan_rules_path = os.path.join(models_dir, 'optimal_loan_rules.joblib')

        PREPROCESSING_PIPELINE_COMPONENTS = joblib.load(preprocessing_pipeline_path)
        SCORE_PARAMS = joblib.load(score_params_path)
        LOAN_RULES = joblib.load(loan_rules_path)
        print("API: Preprocessing pipeline components, score parameters, and loan rules loaded successfully.")

    except Exception as e:
        print(f"API: Error loading models or components at startup: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to load models at startup. Ensure 'mlflow ui' is running and models are registered. Error: {e}")


@app.post("/predict_credit_risk", response_model=list[PredictionResponse])
async def predict_credit_risk(transactions: list[TransactionData]):
    """
    Receives a list of raw transaction data, preprocesses it,
    predicts credit risk probability, assigns a credit score,
    and recommends optimal loan terms for each transaction.
    """
    if PREPROCESSING_PIPELINE_COMPONENTS is None or RISK_MODEL is None or \
       SCORE_PARAMS is None or LOAN_RULES is None:
        raise HTTPException(status_code=503, detail="Models or preprocessing components not loaded. Server is starting up or failed to load them.")

    try:
        # Convert list of Pydantic models to Pandas DataFrame
        raw_df = pd.DataFrame([t.model_dump() for t in transactions])

        # --- Apply Preprocessing Pipeline (features only for prediction) ---
        # The sequential application of transformers, bypassing HighRiskProxyGenerator
        # This logic is copied from predict.py to avoid circular imports and ensure clarity.
        
        # Make a copy to avoid modifying original input
        df_for_processing = raw_df.copy()

        temp_df_intermediate = PREPROCESSING_PIPELINE_COMPONENTS['feature_extractor'].transform(df_for_processing)
        temp_df_intermediate = PREPROCESSING_PIPELINE_COMPONENTS['aggregate_features'].transform(temp_df_intermediate)
        temp_df_intermediate = PREPROCESSING_PIPELINE_COMPONENTS['rfm_calculator'].transform(temp_df_intermediate)
        temp_df_intermediate = PREPROCESSING_PIPELINE_COMPONENTS['kmeans_clusterer'].transform(temp_df_intermediate)
        
        # Apply the column remover that was fitted during training (to remove IDs, etc.)
        temp_df_intermediate = PREPROCESSING_PIPELINE_COMPONENTS['column_dropper_ids_fraud'].transform(temp_df_intermediate)

        # Apply WOE transformation
        X_processed_woe = PREPROCESSING_PIPELINE_COMPONENTS['woe_transformer'].transform(temp_df_intermediate)

        # Apply final numerical scaling
        X_processed_array = PREPROCESSING_PIPELINE_COMPONENTS['final_numerical_scaler'].transform(X_processed_woe)
        
        # The MLflow pyfunc model expects a DataFrame, so convert back.
        # Need to ensure correct column names for the model.
        # The `X_processed_woe.columns` should contain the correct feature names.
        X_processed_df_for_model = pd.DataFrame(X_processed_array, columns=X_processed_woe.columns)

        # --- Predict Risk Probability ---
        # The MLflow pyfunc model's predict method
        # For sklearn models, predict_proba is usually directly available.
        # For pyfunc, you might need to inspect its underlying model type or assume a common interface.
        # If the MLflow model is an sklearn classifier, .predict_proba() should work.
        # If it's a generic pyfunc model, its `predict` method typically returns scores or probabilities.
        # Assuming our sklearn classifier was logged, `predict_proba` is available.
        if hasattr(RISK_MODEL.unwrap_python_model(), 'predict_proba'):
            risk_probabilities = RISK_MODEL.unwrap_python_model().predict_proba(X_processed_df_for_model)[:, 1]
        else:
            # Fallback if predict_proba is not directly exposed by pyfunc wrapper
            # This might require custom prediction logic within the pyfunc model's _predict.py
            raise NotImplementedError("MLflow model does not expose predict_proba. Custom handling needed.")

        # --- Calculate Credit Scores and Loan Recommendations ---
        results_list = []
        for i, pd_value in enumerate(risk_probabilities):
            credit_score = calculate_credit_score(pd_value, SCORE_PARAMS)
            recommended_risk_tier, recommended_amount_range, recommended_duration_months = recommend_loan_terms(pd_value, LOAN_RULES)
            
            # Get original transaction/customer IDs for the response
            original_transaction = transactions[i]
            
            results_list.append(PredictionResponse(
                TransactionId=original_transaction.TransactionId,
                AccountId=original_transaction.AccountId,
                CustomerId=original_transaction.CustomerId, # Add CustomerId to response
                predicted_risk_probability=float(pd_value), # Ensure float type for Pydantic
                credit_score=credit_score,
                recommended_risk_tier=recommended_risk_tier,
                recommended_amount_range=recommended_amount_range,
                recommended_duration_months=recommended_duration_months
            ))
        
        return results_list

    except Exception as e:
        print(f"API: Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error: {str(e)}")

if __name__ == "__main__":
    # To run the API:
    # 1. Ensure all models are trained and registered in MLflow.
    #    Run `python src/data_processing.py` then `python src/train.py`.
    # 2. Make sure MLflow tracking server is running: `mlflow ui` in a separate terminal.
    # 3. Navigate to the 'credit-risk-model/' directory in your terminal.
    # 4. Run: uvicorn src.api.main:app --reload --port 8000
    # Then access at http://127.0.0.1:8000/docs for Swagger UI.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

