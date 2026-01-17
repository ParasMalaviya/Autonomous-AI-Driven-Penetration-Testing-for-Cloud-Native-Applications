import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'Data')
STRUCTURED_PATH = os.path.join(DATA_DIR, 'structured_subset.csv')
X_PATH = os.path.join(DATA_DIR, 'X_train.csv')
Y_PATH = os.path.join(DATA_DIR, 'y_train.csv')

def load_structured_or_xy():
    if os.path.exists(STRUCTURED_PATH):
        df = pd.read_csv(STRUCTURED_PATH)
        if 'CWE_ID' not in df.columns:
            raise RuntimeError('structured_subset.csv missing CWE_ID column')
        y = df['CWE_ID'].astype(str)
        X = df.drop(columns=['CWE_ID'])
        return X, y
    X = pd.read_csv(X_PATH)
    y_df = pd.read_csv(Y_PATH)
    y = y_df['label'] if 'label' in y_df.columns else y_df.iloc[:, 0]
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

def main():
    X, y = load_structured_or_xy()
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        map_path = os.path.join(DATA_DIR, 'cwe_label_map_metadata.csv')
        pd.DataFrame({'label': list(range(len(le.classes_))), 'CWE_ID': le.classes_}).to_csv(map_path, index=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pre, cat_cols, num_cols = build_preprocessor(X)
    pre.fit(X_train)
    X_train_t = pre.transform(X_train)
    X_test_t = pre.transform(X_test)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_t, y_train, test_size=0.2, random_state=42, stratify=y_train)
    if hasattr(X_tr, 'toarray'):
        X_tr = X_tr.toarray()
        X_val = X_val.toarray()
        X_test_arr = X_test_t.toarray()
    else:
        X_test_arr = np.asarray(X_test_t)
    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_test_arr)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 6))
    plot_importance(model, max_num_features=20)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()