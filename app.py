from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved ML pipeline
model = joblib.load('model.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
  prediction_text = None
  probability_text = None
  is_diabetic = None

  if request.method == 'POST':
    try:
      # Extract the 7 features in the exact order used during training
      features = [
          float(request.form['Pregnancies']),
          float(request.form['Glucose']),
          float(request.form['BloodPressure']),
          float(request.form['SkinThickness']),
          float(request.form['BMI']),
          float(request.form['DiabetesPedigreeFunction']),
          float(request.form['Age']),
      ]

      # Replace 0s with np.nan for clinical columns (matching training preprocessing)
      # Indices: 1=Glucose, 2=BloodPressure, 3=SkinThickness, 4=BMI
      for idx in [1, 2, 3, 4]:
        if features[idx] == 0:
          features[idx] = np.nan

      input_data = np.array([features])
      prediction = model.predict(input_data)[0]
      probability = model.predict_proba(input_data)[0][1] * 100

      if prediction == 1:
        is_diabetic = True
        prediction_text = 'High Risk: Diabetic'
      else:
        is_diabetic = False
        prediction_text = 'Low Risk: Non-Diabetic'

      probability_text = f'Estimated Probability: {probability:.1f}%'

    except Exception as e:
      prediction_text = f'Error processing input: {str(e)}'

  return render_template(
      'index.html',
      prediction_text=prediction_text,
      probability_text=probability_text,
      is_diabetic=is_diabetic,
  )


if __name__ == '__main__':
  app.run(debug=True)