# tests/test_data_processing.py

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_processing import RFMCalculator, KMeansClusterer, HighRiskProxyGenerator

# Helper function for dummy data creation (can be reused across tests)
def create_dummy_transactions_df():
    data = {
        'TransactionId': [f'T{i}' for i in range(10)],
        'AccountId': ['A1', 'A1', 'A2', 'A1', 'A3', 'A2', 'A1', 'A3', 'A4', 'A4'],
        'CustomerId': ['C1', 'C1', 'C2', 'C1', 'C3', 'C2', 'C1', 'C3', 'C4', 'C4'],
        'TransactionStartTime': [
            '2023-01-01', '2023-01-05', '2023-01-10', '2023-01-06', '2023-02-01',
            '2023-01-12', '2023-01-07', '2023-02-05', '2023-03-01', '2023-03-02'
        ],
        'Value': [100.0, 50.0, 200.0, 75.0, 300.0, 150.0, 25.0, 100.0, 500.0, 50.0],
        'FraudResult': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # One fraud transaction for C2
        'ProductId': ['P1', 'P2', 'P1', 'P3', 'P2', 'P1', 'P4', 'P3', 'P1', 'P2'] # Required by AggregateFeatures
    }
    df = pd.DataFrame(data)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

# Unit Test 1: Test RFMCalculator correctness
def test_rfm_calculator():
    df = create_dummy_transactions_df()
    
    # Manually define snapshot date for deterministic testing
    snapshot_date = pd.to_datetime('2023-03-03') # One day after latest transaction in dummy data

    rfm_calc = RFMCalculator(snapshot_date=snapshot_date)
    rfm_calc.fit(df) # Fit sets the internal snapshot date
    transformed_df = rfm_calc.transform(df)

    
    # Get RFM values from the transformed DataFrame (unique per customer)
    actual_rfm = transformed_df[['CustomerId', 'Recency', 'Frequency', 'Monetary']].drop_duplicates().set_index('CustomerId')

    assert actual_rfm.loc['C1', 'Recency'] == 55
    assert actual_rfm.loc['C1', 'Frequency'] == 4
    assert np.isclose(actual_rfm.loc['C1', 'Monetary'], 250.0)

    assert actual_rfm.loc['C2', 'Recency'] == 49
    assert actual_rfm.loc['C2', 'Frequency'] == 2
    assert np.isclose(actual_rfm.loc['C2', 'Monetary'], 350.0)

    assert actual_rfm.loc['C3', 'Recency'] == 26
    assert actual_rfm.loc['C3', 'Frequency'] == 2
    assert np.isclose(actual_rfm.loc['C3', 'Monetary'], 400.0)

    assert actual_rfm.loc['C4', 'Recency'] == 1
    assert actual_rfm.loc['C4', 'Frequency'] == 2
    assert np.isclose(actual_rfm.loc['C4', 'Monetary'], 550.0)

    print("RFMCalculator test passed!")


# Unit Test 2: Test HighRiskProxyGenerator functionality
def test_high_risk_proxy_generator():
    df = create_dummy_transactions_df()
    
    # First, apply RFM and KMeans clustering as HighRiskProxyGenerator depends on them
    snapshot_date = pd.to_datetime('2023-03-03')
    rfm_calc = RFMCalculator(snapshot_date=snapshot_date)
    df_rfm = rfm_calc.fit_transform(df)

    kmeans_cluster = KMeansClusterer(n_clusters=3, random_state=42)
    df_clustered = kmeans_cluster.fit_transform(df_rfm) # This adds 'rfm_cluster'

    # Now, test HighRiskProxyGenerator
    high_risk_gen = HighRiskProxyGenerator(n_clusters=3, random_state=42)
    high_risk_gen.fit(df_clustered) # Fit to identify the high-risk cluster
    
    transformed_df = high_risk_gen.transform(df_clustered) # Transform to add 'is_high_risk'

    # Verify 'is_high_risk' column is created and has correct type
    assert 'is_high_risk' in transformed_df.columns
    assert transformed_df['is_high_risk'].dtype == np.int32 # Or int64

    identified_high_risk_cluster = high_risk_gen.high_risk_cluster_id
    
    # Get unique customer data to check 'is_high_risk'
    customer_risk_status = transformed_df[['CustomerId', 'is_high_risk']].drop_duplicates().set_index('CustomerId')

    # Example Check: If C4 is the lowest Recency (most recent), it should be 0 unless it has fraud.
    # C4 has no fraud and is most recent, so it should be 0.
    assert customer_risk_status.loc['C4', 'is_high_risk'] == 0

    # C2 has FraudResult=1, so it should be identified as high risk regardless of RFM cluster
    assert customer_risk_status.loc['C2', 'is_high_risk'] == 1

    # For other customers (C1, C3), their risk depends on whether their cluster was chosen as high-risk.
    # This check is more complex as cluster IDs are arbitrary.
    # Instead, we can verify that at least one customer is flagged high_risk IF one of the conditions is met.
    assert transformed_df['is_high_risk'].sum() > 0 # At least C2 should be high risk due to fraud.

    # Also check that 'rfm_cluster', 'Recency', 'Frequency', 'Monetary' are dropped (due to ColumnRemover or HighRiskProxyGenerator)
    # Note: HighRiskProxyGenerator itself drops RFM columns after use.
    assert 'rfm_cluster' not in transformed_df.columns
    assert 'Recency' not in transformed_df.columns
    assert 'Frequency' not in transformed_df.columns
    assert 'Monetary' not in transformed_df.columns

    print("HighRiskProxyGenerator test passed!")


