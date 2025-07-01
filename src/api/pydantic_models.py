# src/api/pydantic_models.py

from pydantic import BaseModel, Field
from typing import Tuple

class TransactionData(BaseModel):
    """
    Schema for a single transaction data record sent to the API.
    Matches the raw input data fields.
    """
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: str # Expecting ISO format string (e.g., "YYYY-MM-DD HH:MM:SS")
    PricingStrategy: str
    FraudResult: int # Although target-related, it's part of the raw input data


class PredictionResponse(BaseModel):
    """
    Schema for the prediction response from the API.
    Includes predicted risk, score, and loan recommendations.
    """
    TransactionId: str = Field(..., description="Unique transaction identifier")
    AccountId: str = Field(..., description="Unique customer account identifier")
    CustomerId: str = Field(..., description="Unique customer identifier")
    predicted_risk_probability: float = Field(..., description="Predicted Probability of Default (0-1)")
    credit_score: int = Field(..., description="Assigned Credit Score (e.g., 300-850)")
    recommended_risk_tier: str = Field(..., description="Categorized risk tier (e.g., Very Low Risk, High Risk)")
    recommended_amount_range: Tuple[float, float] = Field(..., description="Recommended loan amount range [min, max]")
    recommended_duration_months: Tuple[int, int] = Field(..., description="Recommended loan duration range in months [min, max]")

