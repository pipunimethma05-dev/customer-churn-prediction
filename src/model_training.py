import sys
import os
sys.path.append(os.path.dirname(__file__))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from data_preprocessing import preprocess_data


def train():

    print("⏳ Loading and preprocessing data...")
    X, y, scaler, feature_names = preprocess_data('../data/churn_data.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # ── Compare 3 Models ──────────────────────────────────
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    print("\n" + "=" * 55)
    print("  MODEL COMPARISON — 5-Fold Cross Validation ROC-AUC")
    print("=" * 55)
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train,
                                 cv=5, scoring='roc_auc')
        print(f"  {name:25s}: {scores.mean():.4f} ± {scores.std():.4f}")

    # ── Train Best Model ──────────────────────────────────
    print("\n⏳ Training best model: Random Forest...")
    best_model = RandomForestClassifier(n_estimators=100, random_state=42)
    best_model.fit(X_train, y_train)

    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(classification_report(y_test, y_pred,
                                target_names=['No Churn', 'Churn']))
    print(f"  ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    # ── Confusion Matrix ──────────────────────────────────
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('../confusion_matrix.png', dpi=150)
    plt.show()
    print("✅ Saved: confusion_matrix.png")

    # ── ROC Curve ─────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score   = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('../roc_curve.png', dpi=150)
    plt.show()
    print("✅ Saved: roc_curve.png")

    # ── Feature Importance ────────────────────────────────
    feat_imp = pd.Series(best_model.feature_importances_,
                         index=feature_names).sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
    plt.title('Top 10 Important Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../feature_importance.png', dpi=150)
    plt.show()
    print("✅ Saved: feature_importance.png")

    # ── Save Model ────────────────────────────────────────
    joblib.dump(best_model, '../model.pkl')
    joblib.dump(scaler,     '../scaler.pkl')
    print("\n🎉 Training complete!")
    print("✅ model.pkl saved")
    print("✅ scaler.pkl saved")


if __name__ == '__main__':
    train()