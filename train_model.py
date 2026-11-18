import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
data = pd.read_csv('diabetes.csv')

# 2. Preprocessing (matching your Jupyter Notebook)
data.drop('Insulin', axis=1, inplace=True)

zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
for col in zero_columns:
  data.loc[data[col] == 0, col] = np.nan

# 3. Split features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Build Pipeline (Imputer + Scaler + Classifier)
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)),
])

# 5. Train & Evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f'Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')

# 6. Save the trained pipeline
joblib.dump(pipeline, 'model.pkl')
print('Saved model successfully as model.pkl')