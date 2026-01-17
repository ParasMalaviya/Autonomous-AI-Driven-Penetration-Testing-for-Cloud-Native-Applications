# AI Penetration Testing & CVE Classification - Results Documentation

This document provides a comprehensive overview of all files, code, and actual results contained in this project folder.

**Base Directory**: `drive-download-20260116T113052Z-1-001/`

---

## 📁 Project Structure Overview

```
drive-download-20260116T113052Z-1-001/
├── main.py                          # Main CVE classification script
├── results.txt                      # CVE classification results
├── confusion_matrix.png             # Confusion matrix visualization
├── Figure_1.png - Figure_5.png      # Additional analysis figures
├── CVEfixes_v1.0.8/                 # Vulnerability dataset
├── LO2/                             # AI Penetration Testing Workflow
│   ├── main_workflow.py             # Main workflow script
│   ├── data/                        # Input log data
│   └── outputs/                     # All output results
└── ai_pen_test_all_outputs/         # Archive of all outputs
    ├── CVEfixes_v1.0.8/Data/        # Processed CVE data
    └── LO2/outputs/                 # LO2 workflow outputs
```

---

## 🤖 MODELS USED IN THIS PROJECT

### Model 1: XGBoost Classifier (CVE Classification)

| Property | Value |
|----------|-------|
| **Source File** | `main.py` (Root Directory) |
| **Full Path** | `drive-download-20260116T113052Z-1-001/main.py` |
| **Code Line** | Line 56 |
| **Purpose** | Multi-class classification of CVE vulnerabilities by CWE_ID |

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `objective` | `multi:softprob` | Multi-class probability output |
| `eval_metric` | `mlogloss` | Multi-class log loss |
| `max_depth` | 5 | Maximum tree depth |
| `learning_rate` | 0.1 | Step size shrinkage |
| `n_estimators` | 100 | Number of boosting rounds |
| `random_state` | 42 | Reproducibility seed |

**Code Reference**:
```python
model = XGBClassifier(
    objective='multi:softprob', 
    eval_metric='mlogloss', 
    max_depth=5, 
    learning_rate=0.1, 
    n_estimators=100, 
    random_state=42
)
```

---

### Model 2: Random Forest Classifier (Anomaly Detection)

| Property | Value |
|----------|-------|
| **Source File** | `main_workflow.py` |
| **Full Path** | `drive-download-20260116T113052Z-1-001/LO2/main_workflow.py` |
| **Code Line** | Line 99 |
| **Purpose** | Binary classification for anomaly detection in service logs |

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `n_estimators` | 300 | Number of trees in the forest |
| `random_state` | 42 | Reproducibility seed |
| `n_jobs` | -1 | Use all CPU cores |
| `class_weight` | `balanced_subsample` | Handle class imbalance |

**Code Reference**:
```python
RandomForestClassifier(
    n_estimators=300, 
    random_state=42, 
    n_jobs=-1, 
    class_weight='balanced_subsample'
)
```

---

### Model 3: Gradient Boosting Classifier (Anomaly Detection)

| Property | Value |
|----------|-------|
| **Source File** | `main_workflow.py` |
| **Full Path** | `drive-download-20260116T113052Z-1-001/LO2/main_workflow.py` |
| **Code Line** | Line 100 |
| **Purpose** | Binary classification for anomaly detection (alternative model) |

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `random_state` | 42 | Reproducibility seed |
| *(default)* | `n_estimators=100` | Number of boosting stages |
| *(default)* | `learning_rate=0.1` | Shrinkage parameter |
| *(default)* | `max_depth=3` | Maximum tree depth |

**Code Reference**:
```python
GradientBoostingClassifier(random_state=42)
```

---

### Model 4: Q-Learning (Reinforcement Learning for Penetration Testing)

| Property | Value |
|----------|-------|
| **Source File** | `main_workflow.py` |
| **Full Path** | `drive-download-20260116T113052Z-1-001/LO2/main_workflow.py` |
| **Code Lines** | Lines 151-172 |
| **Environment Class** | `Lo2PenTestEnv` (Lines 129-149) |
| **Purpose** | Train an agent to optimize penetration testing actions |

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate (α)` | 0.1 | Q-value update step size |
| `epsilon (ε)` | 0.2 | Exploration probability |
| `action_space` | 6 actions | Number of possible actions |
| `max_steps` | 200 | Maximum training steps |
| `episode_length` | 50 | Steps until episode terminates |

**Q-Learning Update Rule**:
```python
q[a] = q[a] + 0.1 * (r - q[a])  # Simplified Q-update
```

---

## 📊 SECTION 1: CVE Classification Results (Root Directory)

### Source Code File

| Property | Details |
|----------|---------|
| **File Name** | `main.py` |
| **Location** | Root Directory |
| **Full Path** | `drive-download-20260116T113052Z-1-001/main.py` |
| **Purpose** | XGBoost-based multi-class classifier for CVE classification by CWE_ID |

**Technology Stack**:
| Library | Purpose |
|---------|---------|
| pandas | Data manipulation |
| numpy | Numerical operations |
| scikit-learn | Train-test split, metrics, label encoding |
| XGBoost | Gradient boosting classifier |
| seaborn | Heatmap visualization |
| matplotlib | Plotting |

**Train-Test Split Configuration**:
| Parameter | Value |
|-----------|-------|
| Test Size | 20% |
| Random State | 42 |
| Stratification | Enabled (when possible) |

---

### Results Output File

| Property | Details |
|----------|---------|
| **File Name** | `results.txt` |
| **Location** | Root Directory |
| **Full Path** | `drive-download-20260116T113052Z-1-001/results.txt` |
| **Generated By** | `main.py` |

**Actual Results - CVE Classification**:

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **43.33%** |
| **Total Test Samples** | 1,260 |
| **Total Classes** | 203 |
| **Features Used** | 42 |
| **Macro Average Precision** | 0.4901 |
| **Macro Average Recall** | 0.3530 |
| **Macro Average F1-Score** | 0.3808 |
| **Weighted Average Precision** | 0.4990 |
| **Weighted Average Recall** | 0.4333 |
| **Weighted Average F1-Score** | 0.4157 |

**Top Performing Classes** (F1-Score = 1.0):
- Class 30, 33, 37, 58, 59, 60, 109, 116, 117, 118, 121, 123, 145, 147

---

### Confusion Matrix Visualization

| Property | Details |
|----------|---------|
| **File Name** | `confusion_matrix.png` |
| **Location** | Root Directory |
| **Full Path** | `drive-download-20260116T113052Z-1-001/confusion_matrix.png` |
| **Generated By** | `main.py` (function: `save_confusion_matrix()`) |
| **Description** | Heatmap of predicted vs actual CWE classes |

---

## 📊 SECTION 2: LO2 - AI Penetration Testing Workflow

### Location: `drive-download-20260116T113052Z-1-001/LO2/`

### Source Code File

| Property | Details |
|----------|---------|
| **File Name** | `main_workflow.py` |
| **Location** | `LO2/` subfolder |
| **Full Path** | `drive-download-20260116T113052Z-1-001/LO2/main_workflow.py` |
| **Purpose** | Complete AI-enhanced penetration testing workflow |

**Workflow Components**:

| Component | Function | Lines | Description |
|-----------|----------|-------|-------------|
| `gen_synth_logs()` | Data Generation | 23-56 | Generate 5000 synthetic service logs |
| `parse_logs_to_subset()` | Feature Engineering | 58-91 | Extract security features from logs |
| `train_supervised()` | ML Training | 93-127 | Train RF and GB classifiers |
| `Lo2PenTestEnv` | RL Environment | 129-149 | Penetration testing environment |
| `train_rl()` | RL Training | 151-172 | Q-learning training loop |
| `evaluation()` | Evaluation | 174-191 | Compare Manual/Auto/AI testing |
| `governance()` | Governance | 193-205 | AI safety guardrails |

---

### Subfolder: `LO2/data/`
**Full Path**: `drive-download-20260116T113052Z-1-001/LO2/data/`

| File Name | Full Path | Size | Description |
|-----------|-----------|------|-------------|
| `logs_serviceA.json` | `drive-download-20260116T113052Z-1-001/LO2/data/logs_serviceA.json` | 478 KB | Synthetic service logs (JSONL format) |
| `logs_serviceB.json` | `drive-download-20260116T113052Z-1-001/LO2/data/logs_serviceB.json` | 496 KB | Synthetic service logs (JSONL format) |

**Log Entry Schema**:
```json
{
  "service": "serviceA",
  "ip": "10.0.0.x",
  "endpoint": "/login|/orders|/search|/admin|/health|/metrics",
  "status": 200|201|400|401|403|404|500|502|503,
  "user_agent": "Mozilla/5.0|curl/8.0|Python-requests/2.31|Go-http-client/1.1",
  "port": 80|443|8080|3000|5000,
  "timestamp": <unix_timestamp>,
  "session_id": "s<random_6_digits>",
  "is_anomaly": 0|1
}
```

---

### Subfolder: `LO2/outputs/`
**Full Path**: `drive-download-20260116T113052Z-1-001/LO2/outputs/`

---

#### Output File: `supervised_results.txt`

| Property | Details |
|----------|---------|
| **File Name** | `supervised_results.txt` |
| **Location** | `LO2/outputs/` subfolder |
| **Full Path** | `drive-download-20260116T113052Z-1-001/LO2/outputs/supervised_results.txt` |
| **Generated By** | `main_workflow.py` → `train_supervised()` function |

**Actual Results - Random Forest (RF) Model**:

| Metric | Class 0 (Normal) | Class 1 (Anomaly) |
|--------|------------------|-------------------|
| Precision | 0.9372 | 0.2727 |
| Recall | 0.9655 | 0.1667 |
| F1-Score | 0.9512 | 0.2069 |
| Support | 232 | 18 |

| Overall Metric | Value |
|----------------|-------|
| **Accuracy** | **90.80%** |
| Macro Avg F1 | 0.5790 |
| Weighted Avg F1 | 0.8976 |

**Actual Results - Gradient Boosting (GB) Model**:

| Metric | Class 0 (Normal) | Class 1 (Anomaly) |
|--------|------------------|-------------------|
| Precision | 0.9295 | 0.1111 |
| Recall | 0.9655 | 0.0556 |
| F1-Score | 0.9471 | 0.0741 |
| Support | 232 | 18 |

| Overall Metric | Value |
|----------------|-------|
| **Accuracy** | **90.00%** |
| Macro Avg F1 | 0.5106 |
| Weighted Avg F1 | 0.8843 |

---

#### Output File: `evaluation.txt`

| Property | Details |
|----------|---------|
| **File Name** | `evaluation.txt` |
| **Location** | `LO2/outputs/` subfolder |
| **Full Path** | `drive-download-20260116T113052Z-1-001/LO2/outputs/evaluation.txt` |
| **Generated By** | `main_workflow.py` → `evaluation()` function |

**Actual Results - Testing Methodology Comparison**:

| Metric | Manual | Automated | AI-Driven |
|--------|--------|-----------|-----------|
| **Coverage** | 0.45 (45%) | 0.62 (62%) | **0.78 (78%)** |
| **Time-to-Detect (mins)** | 120 | 60 | **40** |
| **Precision** | 0.55 | 0.70 | **0.76** |
| **Explainability (Likert 1-5)** | - | - | 4.0 |

**Key Finding**: AI-driven penetration testing shows **73% improvement in coverage** and **67% faster detection** compared to manual testing.

---

#### Output File: `governance.yaml`

| Property | Details |
|----------|---------|
| **File Name** | `governance.yaml` |
| **Location** | `LO2/outputs/` subfolder |
| **Full Path** | `drive-download-20260116T113052Z-1-001/LO2/outputs/governance.yaml` |
| **Generated By** | `main_workflow.py` → `governance()` function |

**AI Governance Configuration**:
```json
{
  "sandbox": true,
  "rate_limit_per_min": 120,
  "timeout_seconds": 10,
  "rollback_after_cycle": true,
  "logging": {"enabled": true, "level": "INFO"},
  "frameworks": ["NIST AI RMF 2023", "ENISA AI TL 2024", "Responsible AI"]
}
```

---

#### Output File: `rl_training_log.csv`

| Property | Details |
|----------|---------|
| **File Name** | `rl_training_log.csv` |
| **Location** | `LO2/outputs/` subfolder |
| **Full Path** | `drive-download-20260116T113052Z-1-001/LO2/outputs/rl_training_log.csv` |
| **Generated By** | `main_workflow.py` → `train_rl()` function |

**Reinforcement Learning Training Summary**:

| Metric | Value |
|--------|-------|
| Total Training Steps | 51 |
| Initial Reward | 0.19 |
| Peak Reward | 0.48 (Step 2) |
| Final Reward | -0.01 |
| Exploration Rate (ε) | 20% |
| Learning Rate (α) | 0.1 |

---

#### Output File: `rl_reward_trend.png`

| Property | Details |
|----------|---------|
| **File Name** | `rl_reward_trend.png` |
| **Location** | `LO2/outputs/` subfolder |
| **Full Path** | `drive-download-20260116T113052Z-1-001/LO2/outputs/rl_reward_trend.png` |
| **Generated By** | `main_workflow.py` → `train_rl()` function |
| **Description** | Rolling average (window=5) of RL reward values over training steps |

---

#### Output File: `lo2_cm_supervised.png`

| Property | Details |
|----------|---------|
| **File Name** | `lo2_cm_supervised.png` |
| **Location** | `LO2/outputs/` subfolder |
| **Full Path** | `drive-download-20260116T113052Z-1-001/LO2/outputs/lo2_cm_supervised.png` |
| **Generated By** | `main_workflow.py` → `train_supervised()` function |
| **Description** | Confusion matrix heatmap for Random Forest classifier |

---

#### Output File: `lo2_subset.csv`

| Property | Details |
|----------|---------|
| **File Name** | `lo2_subset.csv` |
| **Location** | `LO2/outputs/` subfolder |
| **Full Path** | `drive-download-20260116T113052Z-1-001/LO2/outputs/lo2_subset.csv` |
| **Generated By** | `main_workflow.py` → `parse_logs_to_subset()` function |
| **Size** | 70 KB |
| **Samples** | 1,250 (25% of 5000 generated logs) |

**Features in Dataset**:
| Feature | Description |
|---------|-------------|
| `label` | Target variable (0=Normal, 1=Anomaly) |
| `request_rate` | Request count per endpoint |
| `path_entropy` | Character diversity in endpoint |
| `is_5xx` | Binary flag for 5xx errors |
| `open_port_exposed` | Ports 80/443/8080 exposed |
| `commit_message_length` | Endpoint path length |
| `lines_added` | 1 if status 200/201 |
| `lines_deleted` | 1 if status 500/502/503 |
| `num_files_changed` | Requests per session |
| `num_functions_changed` | Same as is_5xx |
| `repository_age` | Hours since first timestamp |
| `language_browser` | One-hot: Mozilla user agent |
| `language_curl` | One-hot: curl user agent |
| `language_go` | One-hot: Go user agent |
| `language_python` | One-hot: Python user agent |

---

#### Output File: `lo2_ranked_targets.csv`

| Property | Details |
|----------|---------|
| **File Name** | `lo2_ranked_targets.csv` |
| **Location** | `LO2/outputs/` subfolder |
| **Full Path** | `drive-download-20260116T113052Z-1-001/LO2/outputs/lo2_ranked_targets.csv` |
| **Generated By** | `main_workflow.py` → `train_supervised()` function |
| **Size** | 5.5 KB |
| **Description** | Anomaly probability scores sorted in descending order |

---

#### Output File: `lo2_full_results.txt`

| Property | Details |
|----------|---------|
| **File Name** | `lo2_full_results.txt` |
| **Location** | `LO2/outputs/` subfolder |
| **Full Path** | `drive-download-20260116T113052Z-1-001/LO2/outputs/lo2_full_results.txt` |
| **Generated By** | `main_workflow.py` → `main()` function |
| **Description** | Index of all generated output file paths |

---

## 📊 SECTION 3: CVEfixes_v1.0.8 Dataset

### Location: `drive-download-20260116T113052Z-1-001/CVEfixes_v1.0.8/`

**About**: CVEfixes vulnerability dataset v1.0.8 (July 2024) - collected from U.S. National Vulnerability Database (NVD).

### Dataset Statistics:
| Metric | Value |
|--------|-------|
| Total CVEs Covered | 11,873 |
| Vulnerability Fixing Commits | 12,107 |
| Open Source Projects | 4,249 |
| CWE Types | 272 |
| Files Changed | 51,342 |
| Functions Analyzed | 138,974 |

---

### Subfolder: `CVEfixes_v1.0.8/Data/`
**Full Path**: `drive-download-20260116T113052Z-1-001/CVEfixes_v1.0.8/Data/`

---

#### Data File: `cvefixes_subset.csv`

| Property | Details |
|----------|---------|
| **File Name** | `cvefixes_subset.csv` |
| **Location** | `CVEfixes_v1.0.8/Data/` subfolder |
| **Full Path** | `drive-download-20260116T113052Z-1-001/CVEfixes_v1.0.8/Data/cvefixes_subset.csv` |
| **Size** | 1.74 MB |
| **Used By** | `main.py` for CVE classification |
| **Features** | 42 columns |

---

#### Data File: `cwe_label_map.csv`

| Property | Details |
|----------|---------|
| **File Name** | `cwe_label_map.csv` |
| **Location** | `CVEfixes_v1.0.8/Data/` subfolder |
| **Full Path** | `drive-download-20260116T113052Z-1-001/CVEfixes_v1.0.8/Data/cwe_label_map.csv` |

**CWE Label Encoding**:

| CWE ID | Encoded Label | Vulnerability Type |
|--------|---------------|-------------------|
| CWE-125 | 0 | Out-of-bounds Read |
| CWE-190 | 1 | Integer Overflow or Wraparound |
| CWE-22 | 2 | Path Traversal |
| CWE-416 | 3 | Use After Free |
| CWE-476 | 4 | NULL Pointer Dereference |
| CWE-787 | 5 | Out-of-bounds Write |
| CWE-863 | 6 | Incorrect Authorization |
| NVD-CWE-Other | 7 | Other CWE Types |
| NVD-CWE-noinfo | 8 | No CWE Information Available |

---

#### Data File: `X_train.csv`

| Property | Details |
|----------|---------|
| **File Name** | `X_train.csv` |
| **Location** | `CVEfixes_v1.0.8/Data/` subfolder |
| **Full Path** | `drive-download-20260116T113052Z-1-001/CVEfixes_v1.0.8/Data/X_train.csv` |
| **Size** | 6 KB |
| **Samples** | 1,001 entries |
| **Columns** | `Language`, `Patch_Size` |

---

#### Data File: `y_train.csv`

| Property | Details |
|----------|---------|
| **File Name** | `y_train.csv` |
| **Location** | `CVEfixes_v1.0.8/Data/` subfolder |
| **Full Path** | `drive-download-20260116T113052Z-1-001/CVEfixes_v1.0.8/Data/y_train.csv` |
| **Size** | 3 KB |
| **Samples** | 1,001 entries |
| **Column** | `label` (encoded CWE class 0-8) |

---

## 📊 SECTION 4: ai_pen_test_all_outputs Archive

### Location: `drive-download-20260116T113052Z-1-001/ai_pen_test_all_outputs/`

This folder contains **archived copies** of all outputs. Results are identical to the original locations.

---

### Archive Subfolder: `ai_pen_test_all_outputs/LO2/outputs/`
**Full Path**: `drive-download-20260116T113052Z-1-001/ai_pen_test_all_outputs/LO2/outputs/`

| File Name | Also Found In | Description |
|-----------|---------------|-------------|
| `evaluation.txt` | `LO2/outputs/` | Testing methodology comparison |
| `lo2_cm_supervised.png` | `LO2/outputs/` | RF confusion matrix |
| `lo2_full_results.txt` | `LO2/outputs/` | Output file index |
| `lo2_ranked_targets.csv` | `LO2/outputs/` | Ranked anomaly scores |
| `lo2_subset.csv` | `LO2/outputs/` | Processed feature dataset |
| `rl_reward_trend.png` | `LO2/outputs/` | RL training visualization |
| `rl_training_log.csv` | `LO2/outputs/` | RL training rewards |
| `supervised_results.txt` | `LO2/outputs/` | RF and GB results |

---

### Archive Subfolder: `ai_pen_test_all_outputs/CVEfixes_v1.0.8/Data/`
**Full Path**: `drive-download-20260116T113052Z-1-001/ai_pen_test_all_outputs/CVEfixes_v1.0.8/Data/`

| File Name | Also Found In | Description |
|-----------|---------------|-------------|
| `cvefixes_subset.csv` | `CVEfixes_v1.0.8/Data/` | Main CVE dataset |
| `cwe_label_map.csv` | `CVEfixes_v1.0.8/Data/` | CWE label encoding |
| `X_train.csv` | `CVEfixes_v1.0.8/Data/` | Training features |
| `y_train.csv` | `CVEfixes_v1.0.8/Data/` | Training labels |

---

## 📈 COMPLETE OUTPUT FILE REFERENCE TABLE

| Output/Result | Full File Path | Generated By |
|---------------|----------------|--------------|
| CVE Classification Accuracy (43.33%) | `drive-download-20260116T113052Z-1-001/results.txt` | `main.py` |
| CVE Confusion Matrix | `drive-download-20260116T113052Z-1-001/confusion_matrix.png` | `main.py` |
| RF Accuracy (90.80%) | `drive-download-20260116T113052Z-1-001/LO2/outputs/supervised_results.txt` | `LO2/main_workflow.py` |
| GB Accuracy (90.00%) | `drive-download-20260116T113052Z-1-001/LO2/outputs/supervised_results.txt` | `LO2/main_workflow.py` |
| AI vs Manual Comparison | `drive-download-20260116T113052Z-1-001/LO2/outputs/evaluation.txt` | `LO2/main_workflow.py` |
| RL Training Log | `drive-download-20260116T113052Z-1-001/LO2/outputs/rl_training_log.csv` | `LO2/main_workflow.py` |
| RL Reward Trend | `drive-download-20260116T113052Z-1-001/LO2/outputs/rl_reward_trend.png` | `LO2/main_workflow.py` |
| Governance Config | `drive-download-20260116T113052Z-1-001/LO2/outputs/governance.yaml` | `LO2/main_workflow.py` |
| RF Confusion Matrix | `drive-download-20260116T113052Z-1-001/LO2/outputs/lo2_cm_supervised.png` | `LO2/main_workflow.py` |
| Feature Dataset | `drive-download-20260116T113052Z-1-001/LO2/outputs/lo2_subset.csv` | `LO2/main_workflow.py` |
| Ranked Targets | `drive-download-20260116T113052Z-1-001/LO2/outputs/lo2_ranked_targets.csv` | `LO2/main_workflow.py` |
| Service A Logs | `drive-download-20260116T113052Z-1-001/LO2/data/logs_serviceA.json` | `LO2/main_workflow.py` |
| Service B Logs | `drive-download-20260116T113052Z-1-001/LO2/data/logs_serviceB.json` | `LO2/main_workflow.py` |
| CVE Dataset | `drive-download-20260116T113052Z-1-001/CVEfixes_v1.0.8/Data/cvefixes_subset.csv` | External Dataset |
| CWE Label Map | `drive-download-20260116T113052Z-1-001/CVEfixes_v1.0.8/Data/cwe_label_map.csv` | External Dataset |

---

## 📝 Summary of All Models and Results with File Locations

| Model | Source File | Task | Accuracy | Result File |
|-------|-------------|------|----------|-------------|
| **XGBoost** | `main.py` (Root) | CVE Classification | 43.33% | `results.txt` (Root) |
| **Random Forest** | `LO2/main_workflow.py` | Anomaly Detection | 90.80% | `LO2/outputs/supervised_results.txt` |
| **Gradient Boosting** | `LO2/main_workflow.py` | Anomaly Detection | 90.00% | `LO2/outputs/supervised_results.txt` |
| **Q-Learning** | `LO2/main_workflow.py` | RL Pen Testing | N/A | `LO2/outputs/rl_training_log.csv` |

---

*Document generated: January 16, 2026*  
*Project: SST8396 - AI Penetration Testing Analysis*  
*Models: XGBoost, Random Forest, Gradient Boosting, Q-Learning*
