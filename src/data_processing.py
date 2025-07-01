import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from xverse.transformer import WOEEncoder
import os
import joblib
from datetime import datetime

# --- Custom Transformers ---

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts time-based features from 'TransactionStartTime'.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Ensure TransactionStartTime is datetime
        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])

        X_copy['transaction_hour'] = X_copy['TransactionStartTime'].dt.hour
        X_copy['transaction_day_of_week'] = X_copy['TransactionStartTime'].dt.dayofweek
        X_copy['transaction_month'] = X_copy['TransactionStartTime'].dt.month
        X_copy['transaction_year'] = X_copy['TransactionStartTime'].dt.year
        X_copy['transaction_day_of_month'] = X_copy['TransactionStartTime'].dt.day

        # Removed: Drop original datetime column - keep it for other transformers that might need it
        # X_copy = X_copy.drop(columns=['TransactionStartTime'])

        return X_copy

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction-level data to customer-level features and merges them back.
    This helps in creating features that capture customer behavior over all their transactions.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Ensure necessary columns exist for aggregation
        required_cols = ['AccountId', 'Amount', 'Value', 'TransactionId', 'ProductId', 'ChannelId', 'ProductCategory', 'FraudResult']
        if not all(col in X_copy.columns for col in required_cols):
            missing = list(set(required_cols) - set(X_copy.columns))
            raise ValueError(f"AggregateFeatures missing required columns: {missing}")

        # Ensure 'Amount' and 'Value' are numeric
        X_copy['Amount'] = pd.to_numeric(X_copy['Amount'], errors='coerce')
        X_copy['Value'] = pd.to_numeric(X_copy['Value'], errors='coerce')

        customer_agg = X_copy.groupby('AccountId').agg(
            total_transaction_amount=('Amount', 'sum'),
            avg_transaction_amount=('Amount', 'mean'),
            min_transaction_amount=('Amount', 'min'),
            max_transaction_amount=('Amount', 'max'),
            std_transaction_amount=('Amount', 'std'),
            transaction_count=('TransactionId', 'count'),
            unique_products_bought=('ProductId', lambda x: x.nunique()),
            unique_channels_used=('ChannelId', lambda x: x.nunique()),
            unique_categories_bought=('ProductCategory', lambda x: x.nunique()),
            # Aggregate FraudResult to customer level: 1 if any fraud, 0 otherwise
            customer_has_fraud=('FraudResult', 'max')
        ).reset_index()

        # Fill NaNs from std() if a customer has only one transaction
        customer_agg['std_transaction_amount'] = customer_agg['std_transaction_amount'].fillna(0)

        # Merge these customer-level aggregates back to the transaction-level dataframe
        # This will duplicate customer-level features for each transaction of that customer.
        X_processed = pd.merge(X_copy, customer_agg, on='AccountId', how='left')

        return X_processed

class RFMCalculator(BaseEstimator, TransformerMixin):
    """
    Calculates Recency, Frequency, and Monetary (RFM) metrics for each CustomerId.
    Requires 'TransactionStartTime' and 'Value' columns in the input DataFrame.
    """
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date

    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
        if self.snapshot_date is None:
            # Define snapshot date as one day after the latest transaction in the training data
            self.snapshot_date_ = X_copy['TransactionStartTime'].max() + pd.Timedelta(days=1)
        else:
            self.snapshot_date_ = pd.to_datetime(self.snapshot_date)
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])

        # Calculate RFM metrics at the CustomerId level
        rfm_customer = X_copy.groupby('CustomerId').agg(
            Recency=('TransactionStartTime', lambda date: (self.snapshot_date_ - date.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Value', 'sum')
        ).reset_index()

        # Merge RFM back to the original transaction-level DataFrame
        X_processed = pd.merge(X_copy, rfm_customer, on='CustomerId', how='left')

        return X_processed

class KMeansClusterer(BaseEstimator, TransformerMixin):
    """
    Performs K-Means clustering on RFM features to segment customers.
    Adds 'rfm_cluster' and 'customer_id' (if not already present as 'CustomerId')
    """
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.rfm_cols = ['Recency', 'Frequency', 'Monetary'] # Standard RFM column names

    def fit(self, X, y=None):
        # Ensure RFM columns exist for fitting
        if not all(col in X.columns for col in self.rfm_cols):
            raise ValueError(f"KMeansClusterer expects RFM columns: {self.rfm_cols}")

        # Get unique customers and their RFM values for clustering
        customer_rfm_data = X[['CustomerId'] + self.rfm_cols].drop_duplicates(subset=['CustomerId'])

        # Apply a log transformation for skewed RFM features before scaling and clustering
        # Add a small constant to avoid log(0) if any Monetary or Frequency is zero
        customer_rfm_data['Monetary_log'] = np.log1p(customer_rfm_data['Monetary'])
        customer_rfm_data['Frequency_log'] = np.log1p(customer_rfm_data['Frequency'])
        # Recency is often inverse-scaled, or used directly. Keep it as is for now for simple log(x+1) strategy.

        # Scale the log-transformed RFM features
        self.scaled_rfm_ = self.scaler.fit_transform(customer_rfm_data[['Recency', 'Frequency_log', 'Monetary_log']])

        # Fit KMeans
        self.kmeans.fit(self.scaled_rfm_)
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Ensure RFM columns exist for transformation
        if not all(col in X_copy.columns for col in self.rfm_cols):
            raise ValueError(f"KMeansClusterer expects RFM columns: {self.rfm_cols}")

        # Get unique customers and their RFM values
        customer_rfm_data = X_copy[['CustomerId'] + self.rfm_cols].drop_duplicates(subset=['CustomerId'])

        # Apply same log transformation as during fit
        customer_rfm_data['Monetary_log'] = np.log1p(customer_rfm_data['Monetary'])
        customer_rfm_data['Frequency_log'] = np.log1p(customer_rfm_data['Frequency'])

        # Transform using the fitted scaler
        scaled_rfm_transform = self.scaler.transform(customer_rfm_data[['Recency', 'Frequency_log', 'Monetary_log']])

        # Predict clusters
        customer_rfm_data['rfm_cluster'] = self.kmeans.predict(scaled_rfm_transform)

        # Merge cluster IDs back to the original transaction-level DataFrame
        X_processed = pd.merge(X_copy, customer_rfm_data[['CustomerId', 'rfm_cluster']], on='CustomerId', how='left')

        return X_processed

class HighRiskProxyGenerator(BaseEstimator, TransformerMixin):
    """
    Defines the binary 'is_high_risk' target variable based on RFM clusters and other criteria.
    This should be run after RFM calculation and clustering.
    """
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.high_risk_cluster_id = None # To be set during fit based on cluster analysis

    def fit(self, X, y=None):
        X_copy = X.copy()
        if 'rfm_cluster' not in X_copy.columns or 'Recency' not in X_copy.columns or \
           'Frequency' not in X_copy.columns or 'Monetary' not in X_copy.columns:
            raise ValueError("HighRiskProxyGenerator requires 'rfm_cluster', 'Recency', 'Frequency', 'Monetary' columns.")

        # Analyze clusters to identify the high-risk segment
        # High risk is typically characterized by high Recency (not recent), low Frequency, low Monetary
        # Group by cluster and calculate mean RFM values
        cluster_summary = X_copy.groupby('rfm_cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
        print("\n--- RFM Cluster Summary (for High-Risk Proxy Identification) ---")
        print(cluster_summary)

        # Identify the cluster with high Recency and low Frequency/Monetary
        # This logic can be refined. For now, assume cluster with highest Recency and lowest Frequency/Monetary mean is high-risk.
        # If there's a tie, break it with lowest Frequency.
        # This is an illustrative example and should be driven by business logic and deeper EDA.

        # Sort by Recency (desc), then Frequency (asc), then Monetary (asc)
        sorted_clusters = cluster_summary.sort_values(by=['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True])

        # The first cluster in this sorted list is our candidate for high risk
        self.high_risk_cluster_id = sorted_clusters.iloc[0]['rfm_cluster']
        print(f"\nIdentified high-risk cluster ID based on RFM profile: {self.high_risk_cluster_id}")

        return self

    def transform(self, X):
        X_copy = X.copy()
        if 'rfm_cluster' not in X_copy.columns:
            raise ValueError("Input DataFrame must contain 'rfm_cluster' column.")

        # Initialize 'is_high_risk' column
        X_copy['is_high_risk'] = 0

        # Assign 1 to customers in the identified high-risk cluster
        if self.high_risk_cluster_id is not None:
            # We need to map this based on CustomerId, as rfm_cluster is per CustomerId
            # We assume rfm_cluster is already merged to X_copy.
            X_copy.loc[X_copy['rfm_cluster'] == self.high_risk_cluster_id, 'is_high_risk'] = 1
        else:
            print("Warning: high_risk_cluster_id not identified during fit. 'is_high_risk' will be all 0s.")

        
        if 'customer_has_fraud' in X_copy.columns:
             X_copy.loc[X_copy['customer_has_fraud'] == 1, 'is_high_risk'] = 1

        
        X_copy = X_copy.drop(columns=['rfm_cluster', 'Recency', 'Frequency', 'Monetary',
                                      'Monetary_log', 'Frequency_log', 'customer_has_fraud'], errors='ignore')

        return X_copy


class ColumnRemover(BaseEstimator, TransformerMixin):
    """A transformer to remove specified columns."""
    def __init__(self, columns_to_remove):
        self.columns_to_remove = columns_to_remove

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_remove, errors='ignore')


class WOETransformer(BaseEstimator, TransformerMixin):
   
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
        # Changed handle_unknown from 'value' to 'value' for category_encoders.WOEEncoder
        # handle_unknown='value' is the default and appropriate for category_encoders.WOEEncoder
        self.woe_encoder = WOEEncoder(cols=self.categorical_cols, handle_unknown='value')

    def fit(self, X, y):
        # Only fit on the relevant categorical columns and the target
        cols_to_fit = [col for col in self.categorical_cols if col in X.columns]
        if not cols_to_fit:
            print(f"WOETransformer: No specified categorical columns found for WOE encoding from {self.categorical_cols}.")
            return self

        # WOEEncoder expects X as DataFrame and y as Series/array
        # Ensure y is passed correctly for fitting
        self.woe_encoder.fit(X[cols_to_fit], y)
        return self

    def transform(self, X):
        X_copy = X.copy()
        cols_to_transform = [col for col in self.categorical_cols if col in X_copy.columns]
        if not cols_to_transform:
            return X_copy # No columns to transform

        # Transform only the specified categorical columns
        # WOEEncoder.transform returns a DataFrame with WOE values, replacing original columns
        woe_transformed_df = self.woe_encoder.transform(X_copy[cols_to_transform])

        # Replace original categorical columns with their WOE encoded values
        # Ensure that columns are replaced correctly
        for col in cols_to_transform:
             # category_encoders replaces columns in place if cols is specified
            X_copy[col] = woe_transformed_df[col]


        return X_copy

def preprocess_data(df: pd.DataFrame, pipeline_path: str = None, mode: str = 'train'):
   
    # Identify columns that are identifiers or should be kept for other reasons but not transformed
    id_cols = ['TransactionId', 'AccountId', 'SubscriptionId', 'CustomerId', 'BatchId']

    # Store these columns if needed for merging back later (e.g., for API response)
    df_ids = df[id_cols].copy()

    # Ensure all required columns for initial transformers are present
    required_initial_cols = [
        'TransactionId', 'AccountId', 'CustomerId', 'Amount', 'Value', 'TransactionStartTime',
        'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
        'ProductCategory', 'ChannelId', 'PricingStrategy', 'FraudResult', 'BatchId', 'SubscriptionId'
    ]
    if not all(col in df.columns for col in required_initial_cols):
        missing = list(set(required_initial_cols) - set(df.columns))
        raise ValueError(f"Input DataFrame is missing required columns for preprocessing: {missing}")

    # Create a copy for processing to avoid modifying original df
    df_processing = df.copy()

    # Define categorical columns that will be WOE encoded
    categorical_cols_for_woe = [
        'CurrencyCode', 'ProviderId', 'ProductCategory', 'ChannelId', 'PricingStrategy', 'CountryCode'
    ]
    # Ensure CountryCode is treated as object for WOEEncoder if it's numeric in raw
    if 'CountryCode' in df_processing.columns:
        df_processing['CountryCode'] = df_processing['CountryCode'].astype(str)

    
    columns_to_drop_final = id_cols + ['FraudResult', 'ProductId', 'TransactionStartTime']

    # Define intermediate columns generated by RFM, Aggregation, and Clustering that are dropped by HighRiskProxyGenerator
    intermediate_cols_to_drop = ['rfm_cluster', 'Recency', 'Frequency', 'Monetary',
                                 'Monetary_log', 'Frequency_log', 'customer_has_fraud']


    if mode == 'train':
        print("Building and fitting data preprocessing pipeline in 'train' mode...")

        # Apply initial custom transformers sequentially
        # RFMCalculator needs TransactionStartTime, FeatureExtractor needs TransactionStartTime
        # Both should run before TransactionStartTime is dropped by the final ColumnRemover
        rfm = RFMCalculator().fit(df_processing)
        temp_df = rfm.transform(df_processing) # temp_df now has RFM features

        fe = FeatureExtractor().fit(temp_df) # FeatureExtractor now operates on df with RFM and TransactionStartTime
        temp_df = fe.transform(temp_df) # temp_df now has RFM and extracted time features

        ag = AggregateFeatures().fit(temp_df)
        temp_df = ag.transform(temp_df)

        km = KMeansClusterer(n_clusters=3, random_state=42).fit(temp_df)
        temp_df = km.transform(temp_df)

        hr = HighRiskProxyGenerator(n_clusters=3, random_state=42).fit(temp_df)
        temp_df = hr.transform(temp_df) # This creates 'is_high_risk' and drops intermediate columns

        # Separate features (X) and target (y)
        # The target 'is_high_risk' is now in temp_df
        y_train = temp_df['is_high_risk']
        X_train_features = temp_df.drop(columns=['is_high_risk'], errors='ignore') # Features for final processing

        # Apply column remover - this will drop original IDs, FraudResult, ProductId, TransactionStartTime
        cd = ColumnRemover(columns_to_remove=columns_to_drop_final).fit(X_train_features)
        X_train_features = cd.transform(X_train_features)

        # Fit and transform WOEEncoder
        wt = WOETransformer(categorical_cols=categorical_cols_for_woe).fit(X_train_features, y_train) # WOE needs target
        X_for_scaler = wt.transform(X_train_features) # Apply WOE transformation

        # Fit and transform final scaler
        fns = StandardScaler().fit(X_for_scaler) # Fit final scaler
        X_processed_array = fns.transform(X_for_scaler)


        fitted_pipeline_components = {
            'feature_extractor': fe,
            'aggregate_features': ag,
            'rfm_calculator': rfm,
            'kmeans_clusterer': km,
            'high_risk_proxy_generator': hr, # Keep the fitted instance to potentially use its high_risk_cluster_id if needed elsewhere
            'column_dropper_ids_fraud': cd,
            'woe_transformer': wt,
            'final_numerical_scaler': fns
        }

        # Get feature names after all transformations for the output DataFrame
        # The X_for_scaler.columns will give the correct names after WOE and before final scaling.
        final_feature_names = X_for_scaler.columns.tolist()

        X_processed_df = pd.DataFrame(X_processed_array, columns=final_feature_names)
        X_processed_df['is_high_risk'] = y_train.reset_index(drop=True) # Add target back

        # Save the fitted components in a dictionary
        if pipeline_path:
            os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
            joblib.dump(fitted_pipeline_components, pipeline_path)
            print(f"Data preprocessing pipeline components saved to {pipeline_path}")

        return X_processed_df, fitted_pipeline_components

    elif mode == 'predict':
        if not pipeline_path or not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline not found at {pipeline_path}. Run in 'train' mode first.")
        print(f"Loading data preprocessing pipeline components from {pipeline_path}...")
        fitted_pipeline_components = joblib.load(pipeline_path)

        # Apply transformations sequentially for prediction
        # Note: RiskProxyGenerator is NOT used for prediction features.
        # And we don't assume 'is_high_risk' is in input df for prediction.

        # Apply RFM and FeatureExtractor in the correct order, keeping TransactionStartTime
        temp_df_intermediate = df_processing.copy() # Start from original df for prediction
        temp_df_intermediate = fitted_pipeline_components['rfm_calculator'].transform(temp_df_intermediate)
        temp_df_intermediate = fitted_pipeline_components['feature_extractor'].transform(temp_df_intermediate) # temp_df_intermediate now has RFM and extracted time features

        temp_df_intermediate = fitted_pipeline_components['aggregate_features'].transform(temp_df_intermediate)
        # KMeansClusterer is used in predict mode to assign clusters, its output 'rfm_cluster' is an intermediate column.
        temp_df_intermediate = fitted_pipeline_components['kmeans_clusterer'].transform(temp_df_intermediate)

        # Explicitly drop intermediate columns that are generated but not used as final features,
        # to match the column set produced in train mode after HighRiskProxyGenerator.
        temp_df_intermediate = temp_df_intermediate.drop(columns=intermediate_cols_to_drop, errors='ignore')


        # Apply the column remover that was fitted during training (to remove IDs, TransactionStartTime, etc.)
        temp_df_intermediate = fitted_pipeline_components['column_dropper_ids_fraud'].transform(temp_df_intermediate)

        # Apply WOE transformation
        # WOETransformer needs to handle potential new categories in prediction data. handle_unknown='value' is used.
        X_processed_woe = fitted_pipeline_components['woe_transformer'].transform(temp_df_intermediate)

        # Apply final numerical scaling
        X_processed_array = fitted_pipeline_components['final_numerical_scaler'].transform(X_processed_woe)

        # Reconstruct DataFrame with column names
        final_feature_names = X_processed_woe.columns.tolist() # Names after WOE, before final scaler
        X_processed_df = pd.DataFrame(X_processed_array, columns=final_feature_names)

        return X_processed_df, fitted_pipeline_components
    else:
        raise ValueError("Mode must be 'train' or 'predict'.")

if __name__ == '__main__':
    # This block is for testing the data_processing script independently
    raw_data_path = '../data/raw/transactions.csv'
    processed_data_output_path = '../data/processed/processed_transactions.csv'
    pipeline_save_path = '../models/preprocessing_pipeline.joblib'

    # Create dummy raw data if file doesn't exist (same as in EDA for consistency)
    if not os.path.exists(raw_data_path):
        print(f"Creating dummy data at {raw_data_path} for testing...")
        dummy_data = {
            'TransactionId': range(1000),
            'BatchId': [f'B{i//50}' for i in range(1000)],
            'AccountId': [f'A{i%100}' for i in range(1000)],
            'SubscriptionId': [f'S{i%50}' for i in range(1000)],
            'CustomerId': [f'C{i%100}' for i in range(1000)],
            'CurrencyCode': np.random.choice(['KES', 'UGX', 'TZS', 'USD'], 1000, p=[0.5, 0.2, 0.2, 0.1]),
            'CountryCode': np.random.choice([254, 256, 255, 1], 1000, p=[0.5, 0.2, 0.2, 0.1]),
            'ProviderId': np.random.choice(['P1', 'P2', 'P3', 'P4'], 1000),
            'ProductId': np.random.choice([f'Prod{i}' for i in range(20)], 1000), # Higher cardinality
            'ProductCategory': np.random.choice(['CatA', 'CatB', 'CatC', 'CatD'], 1000),
            'ChannelId': np.random.choice(['Web', 'Android', 'IOS', 'PayLater', 'Checkout'], 1000),
            'Amount': np.random.normal(loc=1000, scale=500, size=1000), # Normally distributed amounts
            'Value': np.abs(np.random.normal(loc=1000, scale=500, size=1000)),
            'TransactionStartTime': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.arange(1000), unit='D'),
            'PricingStrategy': np.random.choice(['StrategyX', 'StrategyY', 'StrategyZ', 'StrategyA'], 1000),
            'FraudResult': np.random.choice([0, 1], 1000, p=[0.98, 0.02]) # Highly imbalanced fraud
        }
        # Introduce some missing values for testing imputation
        for col in ['Amount', 'PricingStrategy', 'ProviderId']:
            missing_indices = np.random.choice(dummy_data['TransactionId'], size=int(0.05 * len(dummy_data['TransactionId'])), replace=False)
            if col == 'Amount':
                dummy_data[col][missing_indices] = np.nan
            else:
                dummy_data[col][missing_indices] = None # For object dtypes

        dummy_df = pd.DataFrame(dummy_data)
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        dummy_df.to_csv(raw_data_path, index=False)
        print("Dummy data created for data_processing.py testing.")

    # Load raw data
    raw_df = pd.read_csv(raw_data_path)

    # Train mode: fit and transform
    processed_df_train, trained_pipeline_components = preprocess_data(raw_df, pipeline_save_path, mode='train')

    # Save processed data
    os.makedirs(os.path.dirname(processed_data_output_path), exist_ok=True)
    processed_df_train.to_csv(processed_data_output_path, index=False)
    print(f"Processed data saved to {processed_data_output_path}")

    print("\nProcessed Data Head (Train Mode):")
    print(processed_df_train.head())
    print("\nProcessed Data Info (Train Mode):")
    processed_df_train.info()
    print("\nTarget Variable Distribution (Train Mode):")
    print(processed_df_train['is_high_risk'].value_counts(normalize=True))

    # Predict mode: load and transform
    # Simulate new data for prediction (e.g., from a new batch)
    new_raw_df = pd.read_csv(raw_data_path).sample(20, random_state=42) 
    processed_df_predict, _ = preprocess_data(new_raw_df, pipeline_save_path, mode='predict')

    print("\nProcessed Data Head (Predict Mode):")
    print(processed_df_predict.head())
    print("\nProcessed Data Info (Predict Mode):")
    processed_df_predict.info()