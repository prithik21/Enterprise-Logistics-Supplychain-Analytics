# Enterprise Logistics & Supply Chain Analytics Engine

## Project Overview
This repository contains an end-to-end data pipeline built to simulate and optimize supply chain operations, warehouse processing delays, and fulfillment duration forecasting. Developed to model key processes in **Logistics and Supply Chain Management** and **Business Intelligence**.

## Key Features
- **Synthetic Supply Chain Pipeline:** Generates realistic logistics datasets involving multi-variable shipping factors.
- **Predictive Lead-Time Modeling:** Implements Random Forest Regression to predict order fulfillment lead times with low MAE.
- **Business Process Analytics:** Features metrics tracking for carrier reliability and warehouse bottleneck detection.

## Data Schema & Tech Stack
- **Languages & Frameworks:** Python 3.10+, Pandas, NumPy, Scikit-Learn
- **Concepts:** Time-to-Value Optimization, Process Mining, Operations Analytics

## Usage
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn
python src/logistics_pipeline.py
