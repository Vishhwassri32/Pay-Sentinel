import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_payments(num_records=500):
    """Generates a realistic synthetic payment dataset modeling fintech transaction failures."""
    np.random.seed(42)
    
    payment_methods = ['UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Wallet']
    currencies = ['INR']
    failure_categories = [
        'INSUFFICIENT_FUNDS', 'CARD_ISSUE', 'BANK_DECLINE', 
        'AUTHENTICATION_FAILURE', 'NETWORK_OR_TIMEOUT', 
        'INVALID_PAYMENT_DETAILS', 'RISK_OR_FRAUD', 
        'CUSTOMER_ABANDONMENT', 'UNKNOWN'
    ]
    
    data = []
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(num_records):
        payment_id = f"pay_{10000 + i}"
        amount = round(np.random.exponential(scale=2500) + 150, 2)
        method = np.random.choice(payment_methods, p=[0.45, 0.25, 0.15, 0.10, 0.05])
        timestamp = start_date + timedelta(seconds=np.random.randint(0, 30*86400))
        
        status = np.random.choice(['SUCCESS', 'FAILED'], p=[0.72, 0.28])
        
        if status == 'FAILED':
            category = np.random.choice(failure_categories, p=[0.22, 0.15, 0.18, 0.12, 0.10, 0.08, 0.05, 0.07, 0.03])
            error_codes = {
                'INSUFFICIENT_FUNDS': 'E_INSUFFICIENT_FUNDS',
                'CARD_ISSUE': 'E_CARD_EXPIRED',
                'BANK_DECLINE': 'E_ISSUER_DECLINE',
                'AUTHENTICATION_FAILURE': 'E_3DS_FAILED',
                'NETWORK_OR_TIMEOUT': 'E_GATEWAY_TIMEOUT',
                'INVALID_PAYMENT_DETAILS': 'E_INVALID_CVV',
                'RISK_OR_FRAUD': 'E_FRAUD_SUSPECTED',
                'CUSTOMER_ABANDONMENT': 'E_USER_CANCELLED',
                'UNKNOWN': 'E_GENERIC_ERROR'
            }
            error_code = error_codes.get(category, 'E_UNKNOWN')
            error_desc = f"Transaction failed due to code {error_code}"
            retry_count = np.random.randint(0, 4)
        else:
            category = 'SUCCESS'
            error_code = 'NONE'
            error_desc = 'Transaction successful'
            retry_count = 0
            
        cust_prev_success = np.random.randint(0, 20)
        cust_prev_failures = np.random.randint(0, 5)
        avg_customer_txn = amount * np.random.uniform(0.8, 1.2)
        
        data.append({
            'payment_id': payment_id,
            'amount': amount,
            'currency': 'INR',
            'payment_method': method,
            'status': status,
            'failure_category': category,
            'error_code': error_code,
            'error_description': error_desc,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'hour': timestamp.hour,
            'day_of_week': timestamp.strftime('%A'),
            'customer_previous_success': cust_prev_success,
            'customer_previous_failures': cust_prev_failures,
            'retry_count': retry_count,
            'transaction_frequency': cust_prev_success + cust_prev_failures + 1,
            'average_customer_transaction': round(avg_customer_txn, 2),
            'recoverable': 1 if status == 'FAILED' and category in ['INSUFFICIENT_FUNDS', 'NETWORK_OR_TIMEOUT', 'AUTHENTICATION_FAILURE'] else 0
        })
        
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/payments.csv', index=False)
    return df

if __name__ == "__main__":
    generate_synthetic_payments()