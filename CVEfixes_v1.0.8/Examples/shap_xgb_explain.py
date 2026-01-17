import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'Data')
X_PATH = os.path.join(DATA_DIR, 'X_train.csv')
Y_PATH = os.path.join(DATA_DIR, 'y_train.csv')

# Load data
X = pd.read_csv(X_PATH)
y = pd.read_csv(Y_PATH)
y = y['label'] if 'label' in y.columns else y.iloc[:, 0]
y = y.astype(int)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build preprocessing (categorical + numeric)
cat_cols = [c for c in X.columns if X[c].dtype == 'object']
num_cols = [c for c in X.columns if c not in cat_cols]
pre = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ]
)

# Fit preprocessor and transform
pre.fit(X_train)
X_train_t = pre.transform(X_train)
X_test_t = pre.transform(X_test)

# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_t, y_train)

# Feature names after preprocessing
cat_feat_names = pre.named_transformers_['cat'].get_feature_names_out(cat_cols) if len(cat_cols) else np.array([])
feat_names = np.concatenate([cat_feat_names, np.array(num_cols)])

# SHAP
shap.initjs()
explainer = shap.TreeExplainer(xgb_model)

# Sample up to 1000 rows from X_test for speed
n_sample = min(1000, X_test.shape[0])
X_sample = X_test.sample(n=n_sample, random_state=42)
X_sample_t = pre.transform(X_sample)

# Convert sparse to dense for SHAP plotting
if hasattr(X_sample_t, 'toarray'):
    X_sample_dense = X_sample_t.toarray()
else:
    X_sample_dense = np.asarray(X_sample_t)

# Compute SHAP values (multi-class returns list of arrays)
shap_values = explainer.shap_values(X_sample_dense)

# Summary plot (bar)
plt.figure()
shap.summary_plot(shap_values, X_sample_dense, feature_names=feat_names, plot_type='bar', show=False)
plt.tight_layout()
bar_path = os.path.join(DATA_DIR, 'shap_summary_bar.png')
plt.savefig(bar_path, dpi=150)
plt.show()

# Beeswarm plot
plt.figure()
shap.summary_plot(shap_values, X_sample_dense, feature_names=feat_names, show=False)
plt.tight_layout()
beeswarm_path = os.path.join(DATA_DIR, 'shap_summary_beeswarm.png')
plt.savefig(beeswarm_path, dpi=150)
plt.show()

# Explain one instance using predicted class
idx = 5 if X_test.shape[0] > 5 else 0
row_df = X_test.iloc[[idx]]
row_t = pre.transform(row_df)
row_dense = row_t.toarray() if hasattr(row_t, 'toarray') else np.asarray(row_t)
proba = xgb_model.predict_proba(row_dense)
cls = int(np.argmax(proba, axis=1)[0])

# Force plot (HTML export) for the chosen class
fp = shap.force_plot(explainer.expected_value[cls] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                     shap_values[cls][0] if isinstance(shap_values, list) else shap_values[0],
                     feature_names=feat_names)
force_html = os.path.join(DATA_DIR, 'shap_force_row.html')
shap.save_html(force_html, fp)

# Decision plot for the chosen class
plt.figure()
sv_row = shap_values[cls][0] if isinstance(shap_values, list) else shap_values[0]
shap.decision_plot(explainer.expected_value[cls] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                   sv_row, feat_names, show=False)
dec_path = os.path.join(DATA_DIR, 'shap_decision_row.png')
plt.savefig(dec_path, dpi=150)
plt.show()

print(bar_path)
print(beeswarm_path)
print(force_html)
print(dec_path)