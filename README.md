# ⚡ PaySentinel AI Hub
### Autonomous Fintech Risk & Revenue Recovery Ecosystem

> *An end-to-end web-stack and machine learning application designed to predict payment friction, diagnose root causes via intelligent agents, and recover lost revenue in real-time for payment gateways like Razorpay.*

---

## 🚀 Overview
Online payment failure (due to gateway timeouts, bank downtimes, or false-positive fraud checks) causes massive revenue leakage for merchants every day. **PaySentinel** transforms traditional reactive error handling into an autonomous, proactive recovery workflow. 

Powered by a modular Python backend, a high-performance ML risk classifier, and an interactive multi-page Flask web application, PaySentinel gives operators total visibility and control over financial health.

---

## ✨ Core Features

1. **📊 Executive Recovery Dashboard (`/`)**
   - Real-time aggregation of total transaction volume, failure rates, total lost revenue, and AI-recoverable revenue.
   - Dynamic failure category breakdown table to identify primary leak points.

2. **📋 Live Transaction Risk Scoring (`/predict-page`)**
   - Interactive risk-evaluation form taking custom transaction parameters (amount, payment method code, hour, day code).
   - Instant ML inference delivering success probability confidence scores and dynamic routing recommendations (e.g., triggering 2FA or secondary gateway switches).

3. **🔍 Agentic Transaction Investigation (`/diagnose-page`)**
   - Autonomous root-cause reasoning engine for failed payment IDs.
   - Provides targeted repair actions, priority levels, and transaction recoverability scores.

4. **📥 Executive Report Export**
   - One-click CSV report generation for offline auditing and stakeholder presentations.

---

## 📂 Project Architecture

```text
razorpay-fraud-ai/
│
├── app.py                      # Master Flask Backend & Routes
├── data/
│   └── payments.csv            # Auto-generated synthetic transaction dataset
├── models/                     # Trained ML models folder
│   ├── failure_classifier.pkl
│   └── recovery_model.pkl
├── static/
│   └── css/
│       └── style.css           # Custom styling sheet
├── templates/                  # Multi-page layout templates
│   ├── base.html               # Master navigation framework
│   ├── index.html              # SaaS Hero & Executive Dashboard
│   ├── predict.html            # Risk Scoring Form Interface
│   └── diagnose.html           # Agentic Diagnosis Interface
└── src/                        # Modular backend architecture
    ├── __init__.py
    ├── data_processing.py
    ├── feature_engineering.py
    ├── ml_model.py
    ├── diagnosis_agent.py
    └── recovery_engine.py