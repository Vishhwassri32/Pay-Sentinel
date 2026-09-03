import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_features(df):
    """Encodes categorical variables and extracts features for ML models."""
    df_ml = df.copy()
    
    # Handle missing values
    df_ml.fillna(0, inplace=True)
    
    le_method = LabelEncoder()
    le_error = LabelEncoder()
    le_dow = LabelEncoder()
    
    df_ml['payment_method_enc'] = le_method.fit_transform(df_ml['payment_method'])
    df_ml['error_code_enc'] = le_error.fit_transform(df_ml['error_code'])
    df_ml['day_of_week_enc'] = le_dow.fit_transform(df_ml['day_of_week'])
    
    feature_cols = [
        'amount', 'payment_method_enc', 'hour', 'day_of_week_enc',
        'customer_previous_success', 'customer_previous_failures',
        'retry_count', 'error_code_enc', 'transaction_frequency',
        'average_customer_transaction'
    ]
    
    return df_ml, feature_cols