import sys
sys.path.append('src')

from data_preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

print("⏳ Loading data...")
X, y, scaler, features = preprocess_data('data/churn_data.csv')

print(f"✅ Features found: {features}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("⏳ Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ model.pkl saved!")
print("✅ scaler.pkl saved!")
print(f"✅ Model features: {list(model.feature_names_in_)}")