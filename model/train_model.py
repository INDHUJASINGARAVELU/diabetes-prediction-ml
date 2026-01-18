import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
import joblib

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("../diabd.csv")

print("Dataset loaded")
print("Original class distribution:")
print(data['diabetic'].value_counts())

# -----------------------------
# 2. Encode Gender
# -----------------------------
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})

# -----------------------------
# 3. Handle Missing Values
# -----------------------------
numeric_cols = data.select_dtypes(include=np.number).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# -----------------------------
# 4. Features & Target
# -----------------------------
X = data.drop('diabetic', axis=1)
y = data['diabetic']

# -----------------------------
# 5. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 6. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 7. APPLY SMOTE (KEY STEP)
# -----------------------------
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_scaled, y_train
)

print("\nBalanced class distribution:")
print(pd.Series(y_train_balanced).value_counts())

# -----------------------------
# 8. Train Random Forest
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

rf_model.fit(X_train_balanced, y_train_balanced)

# -----------------------------
# 9. Evaluation
# -----------------------------
y_pred = rf_model.predict(X_test_scaled)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# 10. Save Model & Scaler
# -----------------------------
joblib.dump(rf_model, "diabetes_random_forest_balanced.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nBalanced model saved successfully!")
