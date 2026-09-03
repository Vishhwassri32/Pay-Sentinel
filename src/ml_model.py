import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.data_processing import generate_synthetic_payments
from src.feature_engineering import preprocess_features

def train_models():
    """Trains classification models for failure categorization and recovery prediction."""
    if not os.path.exists('data/payments.csv'):
        generate_synthetic_payments()
        
    df = pd.read_csv('data/payments.csv')
    failed_df = df[df['status'] == 'FAILED'].copy()
    
    if len(failed_df) < 10:
        failed_df = df.copy() # fallback if too few failures
        
    df_ml, feature_cols = preprocess_features(failed_df)
    
    X = df_ml[feature_cols]
    y_category = failed_df['failure_category']
    y_recovery = failed_df['recoverable']
    
    X_train, X_test, y_cat_train, y_cat_test, y_rec_train, y_rec_test = train_test_split(
        X, y_category, y_recovery, test_size=0.2, random_state=42
    )
    
    # Failure Category Classifier
    cat_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cat_model.fit(X_train, y_cat_train)
    
    # Recovery Probability Classifier
    rec_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rec_model.fit(X_train, y_rec_train)
    
    # Evaluate Category Model
    y_cat_pred = cat_model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_cat_test, y_cat_pred),
        'precision': precision_score(y_cat_test, y_cat_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_cat_test, y_cat_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_cat_test, y_cat_pred, average='weighted', zero_division=0)
    }
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(cat_model, 'models/failure_classifier.pkl')
    joblib.dump(rec_model, 'models/recovery_model.pkl')
    
    return cat_model, rec_model, metrics

def load_models():
    if not os.path.exists('models/failure_classifier.pkl') or not os.path.exists('models/recovery_model.pkl'):
        return train_models()[:2]
    cat_model = joblib.load('models/failure_classifier.pkl')
    rec_model = joblib.load('models/recovery_model.pkl')
    return cat_model, rec_model