from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional

class FinancialTransaction(BaseModel):
    """
    Pydantic schema to validate raw transaction data from the Bronze layer.
    """
    transaction_id: str = Field(..., description="Unique identifier for the transaction")
    date: datetime = Field(..., description="Date of the transaction")
    amount: float = Field(..., description="Transaction amount (can be positive for income, negative for expense)")
    merchant_name: str = Field(..., description="Raw merchant name from the bank")
    category: Optional[str] = Field(default="Uncategorized", description= "Transaction category")
    
    # Custom Validatore to ensure amount doesnt have crazy values
    @field_validator('amount')
    def check_amount_range(cls, value):
        if value > 1_000_000 or value < -1_000_000:
            raise ValueError(f"Amount {value} looks like a parsing error. Must be within realistic bounds.")
        return round(value, 2)
        
    
    # Custom Validator: Clean up merchant names
    @field_validator('merchant_name')
    def clean_merchant_name(cls, value):
        return value.strip().upper()