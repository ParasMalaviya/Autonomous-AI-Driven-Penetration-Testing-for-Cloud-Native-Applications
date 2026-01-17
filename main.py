import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

def find_subset():
    root = os.getcwd()
    candidates = [
        os.path.join(root, 'cvefixes_subset.csv'),
        os.path.join(root, 'CVEfixes_v1.0.8', 'Data', 'cvefixes_subset.csv'),
        os.path.join(root, 'Data', 'cvefixes_subset.csv')
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError('cvefixes_subset.csv not found')

def load_data(path):
    df = pd.read_csv(path)
    label_col = 'CWE_ID' if 'CWE_ID' in df.columns else ('cwe_id' if 'cwe_id' in df.columns else None)
    if label_col is None:
        raise RuntimeError('CWE_ID/cwe_id column missing in subset')
    X = df.drop(columns=[label_col])
    y = df[label_col].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le

def ensure_numeric(X):
    for c in X.columns:
        if X[c].dtype == object:
            try:
                X[c] = pd.to_numeric(X[c], errors='raise')
            except Exception:
                X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
    return X

def train_and_eval(X, y):
    X = ensure_numeric(X.copy())
    vc = pd.Series(y).value_counts()
    keep_classes = vc[vc >= 2].index
    mask = pd.Series(y).isin(keep_classes).to_numpy()
    X = X.loc[mask]
    y = y[mask]
    y = LabelEncoder().fit_transform(y)
    strat = y if pd.Series(y).value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
    model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', max_depth=5, learning_rate=0.1, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    return model, (X_train, X_test, y_train, y_test), y_pred, y_proba, acc, report, cm

def save_confusion_matrix(cm, out_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()

def write_results(results_path, acc, report, cm_path, n_classes, n_features):
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write('Classification Report:\n')
        f.write(report + '\n')
        f.write(f'Confusion Matrix Image: {cm_path}\n')
        f.write(f'Classes: {n_classes}\n')
        f.write(f'Features: {n_features}\n')

if __name__ == '__main__':
    subset_path = find_subset()
    X, y, le = load_data(subset_path)
    model, splits, y_pred, y_proba, acc, report, cm = train_and_eval(X, y)
    cm_path = os.path.join(os.getcwd(), 'confusion_matrix.png')
    save_confusion_matrix(cm, cm_path)
    results_path = os.path.join(os.getcwd(), 'results.txt')
    write_results(results_path, acc, report, cm_path, len(le.classes_), X.shape[1])
    print(results_path)