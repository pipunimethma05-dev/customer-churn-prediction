import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Drop customerID
    df.drop('customerID', axis=1, inplace=True)

    # Fix TotalCharges — some rows have blank spaces instead of numbers
    df['TotalCharges'] = df['TotalCharges'].replace(' ', float('nan'))
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Encode target column first
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Encode ALL text (object) columns to numbers BEFORE filling NaN
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Now safely fill any remaining NaN with median (all columns are numbers now)
    df.fillna(df.median(), inplace=True)

    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Confirm no NaN remains
    print(f"✅ NaN check — X: {X.isnull().sum().sum()}, y: {y.isnull().sum()}")
    print(f"✅ Train ready — Rows: {X.shape[0]}, Columns: {X.shape[1]}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    feature_names = X.columns.tolist()

    return X_scaled, y, scaler, feature_names