import numpy as np
import pandas as pd
from src.feature_engineering import preprocess_features
from src.ml_model import load_models

class PaySentinelAgent:
    """Agentic AI system for reasoning over failed payment signals and recommending actions."""
    
    def __init__(self):
        self.cat_model, self.rec_model = load_models()
        
    def diagnose_transaction(self, tx_row, full_df):
        """Performs multi-step agentic investigation on a transaction."""
        df_ml, feature_cols = preprocess_features(full_df)
        match_row = df_ml[df_ml['payment_id'] == tx_row['payment_id']]
        
        if match_row.empty:
            # Fallback single row dataframe processing
            temp_df = pd.DataFrame([tx_row])
            df_ml, feature_cols = preprocess_features(temp_df)
            match_row = df_ml
            
        X_sample = match_row[feature_cols]
        
        # ML Predictions
        predicted_category = self.cat_model.predict(X_sample)[0]
        cat_probs = self.cat_model.predict_proba(X_sample)[0]
        confidence = float(np.max(cat_probs))
        
        rec_prob = float(self.rec_model.predict_proba(X_sample)[0][1]) if hasattr(self.rec_model, 'predict_proba') else 0.65
        
        # Agentic Decision Layer & Explanation
        recommendation, reason, priority = self._generate_recommendation(predicted_category, tx_row)
        
        return {
            "payment_id": tx_row['payment_id'],
            "failure_category": predicted_category,
            "root_cause": self._get_root_cause(predicted_category, tx_row.get('error_code', '')),
            "confidence": round(confidence, 2),
            "recoverability_score": round(rec_prob, 2),
            "recommended_action": recommendation,
            "explanation": reason,
            "priority": priority
        }
        
    def _get_root_cause(self, category, error_code):
        causes = {
            'INSUFFICIENT_FUNDS': 'Customer account balance is lower than the transaction amount.',
            'CARD_ISSUE': 'Card has expired, is blocked, or details were entered incorrectly.',
            'BANK_DECLINE': 'The issuing bank declined the transaction due to internal security policies.',
            'AUTHENTICATION_FAILURE': '3DS Two-Factor Authentication (OTP) failed or timed out.',
            'NETWORK_OR_TIMEOUT': 'Payment gateway timed out while communicating with the banking network.',
            'INVALID_PAYMENT_DETAILS': 'Incorrect CVV, PIN, or VPA handle entered by the user.',
            'RISK_OR_FRAUD': 'Triggered velocity or geographical fraud risk filters.',
            'CUSTOMER_ABANDONMENT': 'User closed the browser window or cancelled the payment prompt.',
            'UNKNOWN': 'Unrecognized error returned by the banking gateway.'
        }
        return causes.get(category, 'Transaction failed due to unclassified gateway error.')
        
    def _generate_recommendation(self, category, tx):
        amount = tx.get('amount', 0)
        method = tx.get('payment_method', 'Card')
        
        if category == 'INSUFFICIENT_FUNDS':
            return (
                f"Offer UPI as an alternative payment method or prompt a retry later.",
                f"The customer has history of successful payments (Avg: ₹{tx.get('average_customer_transaction', 0)}), but current balance failed for ₹{amount}.",
                "HIGH"
            )
        elif category == 'CARD_ISSUE':
            return (
                f"Ask the customer to update their saved payment method or use Net Banking.",
                f"Card-specific error detected ({tx.get('error_code', 'Card Error')}). Immediate update required.",
                "HIGH"
            )
        elif category == 'NETWORK_OR_TIMEOUT':
            return (
                f"Automatically retry the transaction after a short 10-second delay.",
                f"Gateway timeout detected. No funds were debited from customer account.",
                "MEDIUM"
            )
        elif category == 'AUTHENTICATION_FAILURE':
            return (
                f"Trigger an expedited 1-click OTP verification checkout flow.",
                f"Customer aborted or failed 3DS authentication during {method} processing.",
                "HIGH"
            )
        elif category == 'RISK_OR_FRAUD':
            return (
                f"Flag transaction for manual review. Avoid automated retries.",
                f"Unusual transaction parameters detected compared to customer's historical baseline.",
                "CRITICAL"
            )
        else:
            return (
                f"Prompt customer to use an alternative payment channel ({method} alternative).",
                f"General failure category detected with retry count {tx.get('retry_count', 0)}.",
                "MEDIUM"
            )