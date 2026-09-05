from flask import Flask, render_template, request, send_file, jsonify
import os
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.io as pio

from src.data_processing import generate_synthetic_payments
from src.ml_model import load_models
from src.diagnosis_agent import PaySentinelAgent
from src.recovery_engine import RevenueRecoveryEngine

app = Flask(__name__)

if not os.path.exists('data/payments.csv'):
    generate_synthetic_payments(500)

df = pd.read_csv('data/payments.csv')
agent = PaySentinelAgent()
recovery_engine = RevenueRecoveryEngine(df)

try:
    cat_model, rec_model = load_models()
except Exception as e:
    cat_model, rec_model = None, None
    print(f"Warning: Models could not be loaded: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    metrics = recovery_engine.calculate_metrics()
    breakdown_df = recovery_engine.get_revenue_by_category()
    
    # Plotly Bar Chart for Revenue Loss by Category
    fig_cat = px.bar(
        breakdown_df, 
        x='Failure Category', 
        y='Lost Revenue',
        title='Revenue Leakage by Failure Category',
        template='plotly_dark',
        color='Lost Revenue',
        color_continuous_scale='Reds'
    )
    fig_cat.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20, l=20, r=20)
    )
    chart_html_cat = pio.to_html(fig_cat, full_html=False)

    # Plotly Donut Chart for Transaction Health Split
    status_counts = df['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    fig_status = px.pie(
        status_counts, 
        names='Status', 
        values='Count', 
        hole=0.6,
        title='Transaction Health Distribution',
        template='plotly_dark',
        color_discrete_map={'SUCCESS': '#34d399', 'FAILED': '#f87171'}
    )
    fig_status.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20, l=20, r=20)
    )
    chart_html_status = pio.to_html(fig_status, full_html=False)

    return render_template(
        'dashboard.html', 
        metrics=metrics, 
        breakdown=breakdown_df.to_dict(orient='records'),
        chart_cat=chart_html_cat,
        chart_status=chart_html_status
    )

@app.route('/predict-page', methods=['GET', 'POST'])
def predict_page():
    prediction, status_title, status_color, confidence, advice, form_data, error = None, None, None, None, None, None, None

    if request.method == 'POST':
        if cat_model is None:
            error = "Models not found in models/ directory."
        else:
            try:
                amount = float(request.form['amount'])
                payment_method_enc = int(request.form['payment_method_enc'])
                hour = int(request.form['hour'])
                day_of_week_enc = int(request.form['day_of_week_enc'])
                
                input_data = pd.DataFrame([[
                    amount, payment_method_enc, hour, day_of_week_enc,
                    10, 0, 0, 0, 4.5, 2000.0
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
                    status_color = "#34d399"
                    advice = "Transaction parameters look stable. Route normally."
                else:
                    status_title = "High Failure / Risk Detected"
                    status_color = "#f87171"
                    advice = "Recommended Action: Trigger 2FA or route through a secondary gateway."
                
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

@app.route('/diagnose-page')
def diagnose_page():
    failed_txs = df[df['status'] == 'FAILED']['payment_id'].tolist()
    return render_template('diagnose.html', failed_txs=failed_txs)

@app.route('/api/diagnose', methods=['POST'])
def api_diagnose():
    try:
        selected_pay_id = request.form.get('payment_id')
        tx_row_match = df[df['payment_id'] == selected_pay_id]
        if not tx_row_match.empty:
            tx_row = tx_row_match.iloc[0].to_dict()
            diagnosis_result = agent.diagnose_transaction(tx_row, df)
            return jsonify({'success': True, 'diagnosis': diagnosis_result})
        return jsonify({'success': False, 'error': 'Payment ID not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export_report')
def export_report():
    csv_data = df.to_csv(index=False)
    return send_file(
        io.BytesIO(csv_data.encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='paysentinel_executive_report.csv'
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)