import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# Locate CSVs
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'Data')

def find_csv(fname):
    candidates = [
        os.path.join(DATA_DIR, fname),
        os.path.join(ROOT, fname),
        os.path.join(os.getcwd(), fname)
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{fname} not found in {candidates}")

def load_data():
    x_path = find_csv('X_train.csv')
    y_path = find_csv('y_train.csv')
    X = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    if 'label' in y_df.columns:
        y = y_df['label']
    else:
        y = y_df.iloc[:, 0]
    y = y.astype(int)
    return X, y

def build_preprocessor(X):
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ]
    )
    return pre, cat_cols, num_cols

def plot_confusion(cm, classes, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

def plot_f1_bars(scores):
    plt.figure(figsize=(6, 4))
    models = list(scores.keys())
    values = [scores[m] for m in models]
    sns.barplot(x=models, y=values)
    plt.ylabel('F1 Macro')
    plt.ylim(0, 1)
    plt.title('F1 Macro per Model')
    plt.tight_layout()
    plt.show()

def top_features(importances, feature_names, k=10):
    idx = np.argsort(importances)[::-1][:k]
    return [(feature_names[i], float(importances[i])) for i in idx]

def main():
    X, y = load_data()
    pre, cat_cols, num_cols = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Feature names after preprocessing
    pre.fit(X_train)
    cat_feat_names = pre.named_transformers_['cat'].get_feature_names_out(cat_cols) if len(cat_cols) else np.array([])
    feat_names = np.concatenate([cat_feat_names, np.array(num_cols)])

    # Models
    models = {
        'LogReg': Pipeline(steps=[('preprocess', pre), ('clf', LogisticRegression(max_iter=1000))]),
        'RF': Pipeline(steps=[('preprocess', pre), ('clf', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced_subsample'))])
    }
    if XGB_AVAILABLE:
        models['XGB'] = Pipeline(steps=[('preprocess', pre), ('clf', XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1))])

    f1_scores = {}

    for name, pipe in models.items():
        # Train
        pipe.fit(X_train, y_train)
        # Predict
        y_pred = pipe.predict(X_test)
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores[name] = f1
        print(f"=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Macro: {f1:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, digits=4))
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion(cm, classes=sorted(np.unique(y)), title=f"Confusion Matrix - {name}")

        # Feature importance (RF and XGB)
        clf = pipe.named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            tf = top_features(importances, feat_names, k=10)
            print("Top 10 Important Features:")
            for fn, val in tf:
                print(f"{fn}: {val:.4f}")

    # F1 bar chart
    plot_f1_bars(f1_scores)

if __name__ == '__main__':
    main()