# 📊 Customer Churn Prediction

A machine learning project to predict customer churn using the Telco dataset.

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Random Forest, Logistic Regression, Gradient Boosting)
- Matplotlib, Seaborn
- Streamlit

## 📁 Project Structure
customer_churn_prediction/
├── data/churn_data.csv
├── src/
│   ├── eda.py
│   ├── data_preprocessing.py
│   └── model_training.py
├── app.py
├── requirements.txt
└── README.md

## 🚀 How to Run
pip install -r requirements.txt
streamlit run app.py

## 📊 Results
- Best Model: Random Forest
- ROC-AUC Score: ~0.83
- Key Churn Drivers: Tenure, Contract Type, Monthly Charges