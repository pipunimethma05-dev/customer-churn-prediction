import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(filepath):

    df = pd.read_csv(filepath)

    # Fix columns for heatmap
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df_encoded = df.copy()
    df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

    print("=" * 50)
    print("DATASET INFO")
    print("=" * 50)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"\nChurn counts:\n{df['Churn'].value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # ── Plot 1: Churn Distribution ────────────────────────
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df, palette='Set2')
    plt.title('Churn Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('churn_distribution.png', dpi=150)
    plt.show()
    print("✅ Plot saved: churn_distribution.png")

    # ── Plot 2: Churn by Contract Type ────────────────────
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Contract', hue='Churn', data=df, palette='Set1')
    plt.title('Churn by Contract Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('churn_by_contract.png', dpi=150)
    plt.show()
    print("✅ Plot saved: churn_by_contract.png")

    # ── Plot 3: Tenure vs Churn ───────────────────────────
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='tenure', hue='Churn', bins=30, palette='Set1')
    plt.title('Tenure vs Churn', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('tenure_vs_churn.png', dpi=150)
    plt.show()
    print("✅ Plot saved: tenure_vs_churn.png")

    # ── Plot 4: Monthly Charges vs Churn ──────────────────
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set2')
    plt.title('Monthly Charges vs Churn', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('monthly_charges_vs_churn.png', dpi=150)
    plt.show()
    print("✅ Plot saved: monthly_charges_vs_churn.png")

    # ── Plot 5: Correlation Heatmap ───────────────────────
    plt.figure(figsize=(13, 9))
    sns.heatmap(df_encoded.select_dtypes(include='number').corr(),
                annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150)
    plt.show()
    print("✅ Plot saved: correlation_heatmap.png")

    print("\n🎉 EDA Complete! Check your project folder for saved plots.")


# Run this file directly
if __name__ == '__main__':
    run_eda('../data/churn_data.csv')