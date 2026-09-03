from flask import Flask, render_template, request, send_file
import os
import pandas as pd
import numpy as np
import io

# Importing modules from modular src/ architecture
from src.data_processing import generate_synthetic_payments
from src.feature_engineering import preprocess_features
from src.ml_model import load_models
from src.diagnosis_agent import PaySentinelAgent
from src.recovery_engine import RevenueRecoveryEngine

app = Flask(__name__)

# Initialize Dataset, Agent, and Recovery Engine at startup
if not os.path.exists('data/payments.csv'):
    generate_synthetic_payments(500)

df = pd.read_csv('data/payments.csv')
agent = PaySentinelAgent()
recovery_engine = RevenueRecoveryEngine(df)

# Load trained models safely
try:
    cat_model, rec_model = load_models()
except Exception as e:
    cat_model, rec_model = None, None
    print(f"Warning: Models could not be loaded: {e}")


@app.route('/')
def home():
    metrics = recovery_engine.calculate_metrics()
    breakdown_df = recovery_engine.get_revenue_by_category()
    return render_template(
        'index.html', 
        metrics=metrics, 
        breakdown=breakdown_df.to_dict(orient='records')
    )


@app.route('/predict-page', methods=['GET', 'POST'])
def predict_page():
    prediction = None
    status_title, status_color, confidence, advice, form_data = None, None, None, None, None
    error = None

    if request.method == 'POST':
        if cat_model is None:
            error = "Models not found in models/ directory."
        else:
            try:
                amount = float(request.form['amount'])
                payment_method_enc = int(request.form['payment_method_enc'])
                hour = int(request.form['hour'])
                day_of_week_enc = int(request.form['day_of_week_enc'])
                customer_previous_success = int(request.form['customer_previous_success'])
                customer_previous_failures = int(request.form['customer_previous_failures'])
                retry_count = int(request.form['retry_count'])
                error_code_enc = int(request.form['error_code_enc'])
                transaction_frequency = float(request.form['transaction_frequency'])
                average_customer_transaction = float(request.form['average_customer_transaction'])

                input_data = pd.DataFrame([[
                    amount, payment_method_enc, hour, day_of_week_enc,
                    customer_previous_success, customer_previous_failures,
                    retry_count, error_code_enc, transaction_frequency,
                    average_customer_transaction
                ]], columns=[
                    'amount', 'payment_method_enc', 'hour', 'day_of_week_enc',
                    'customer_previous_success', 'customer_previous_failures',
                    'retry_count', 'error_code_enc', 'transaction_frequency',
                    'average_customer_transaction'
                ])

                probabilities = cat_model.predict_proba(input_data)[0]
                confidence = f"{float(np.max(probabilities) * 100):.2f}%"
                pred_class = cat_model.predict(input_data)[0]

                if str(pred_class).upper() in ['SUCCESS', '1', 'LOW_RISK']:
                    status_title = "Low Risk / High Success Probability"
                    status_color = "green"
                    advice = "Transaction parameters look stable. Route normally via the primary gateway channel."
                else:
                    status_title = "High Failure / Risk Detected"
                    status_color = "red"
                    advice = "Recommended Action: Trigger 2FA or route through a secondary secure banking partner."
                
                prediction = True
                form_data = request.form
            except Exception as e:
                error = str(e)

    return render_template(
        'predict.html', 
        prediction=prediction, 
        status_title=status_title, 
        status_color=status_color, 
        confidence=confidence, 
        advice=advice, 
        form_data=form_data, 
        error=error
    )


@app.route('/diagnose-page', methods=['GET', 'POST'])
def diagnose_page():
    failed_txs = df[df['status'] == 'FAILED']['payment_id'].tolist()
    diagnosis_result = None
    selected_id = None

    if request.method == 'POST':
        selected_id = request.form.get('payment_id')
        tx_row_match = df[df['payment_id'] == selected_id]
        if not tx_row_match.empty:
            tx_row = tx_row_match.iloc[0].to_dict()
            diagnosis_result = agent.diagnose_transaction(tx_row, df)

    return render_template(
        'diagnose.html', 
        failed_txs=failed_txs, 
        diagnosis=diagnosis_result, 
        selected_id=selected_id
    )


@app.route('/export_report')
def export_report():
    csv_data = df.to_csv(index=False)
    return send_file(
        io.BytesIO(csv_data.encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='paysentinel_executive_recovery_report.csv'
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)