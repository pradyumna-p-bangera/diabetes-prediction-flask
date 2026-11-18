# Diabetes Prediction Web App

A full-stack machine learning web application built with **Flask** and **Scikit-Learn** that predicts diabetes risk based on clinical diagnostic measurements (such as Glucose, Blood Pressure, BMI, Insulin, and Age).

## Project Structure
- `train_model.py` — Loads the dataset, preprocesses features, trains the classification model, and exports `model.pkl`.
- `app.py` — Flask backend serving the web interface and `/predict` JSON API endpoint.
- `templates/index.html` — Interactive frontend UI for entering patient metrics and viewing predictions.
- `requirements.txt` — Python dependencies required to run the project.

## Input Features
- **Pregnancies** — Number of times pregnant
- **Glucose** — Plasma glucose concentration (mg/dL)
- **Blood Pressure** — Diastolic blood pressure (mm Hg)
- **Skin Thickness** — Triceps skin fold thickness (mm)
- **Insulin** — 2-Hour serum insulin (mu U/ml)
- **BMI** — Body mass index ($\text{weight in kg}/(\text{height in m})^2$)
- **Diabetes Pedigree Function** — Genetic likelihood score based on family history
- **Age** — Age in years

## How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt