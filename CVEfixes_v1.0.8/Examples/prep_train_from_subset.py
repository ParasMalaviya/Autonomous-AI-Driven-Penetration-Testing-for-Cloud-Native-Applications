import os
import pandas as pd

# Input subset and output train files
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'Data')
SUBSET = os.path.join(DATA_DIR, 'cve_dataset_subset.csv')
X_OUT = os.path.join(DATA_DIR, 'X_train.csv')
Y_OUT = os.path.join(DATA_DIR, 'y_train.csv')

def main():
    df = pd.read_csv(SUBSET)
    df = df.dropna(subset=['programming_language', 'patch_size', 'cwe_id'])
    # Features
    X = pd.DataFrame({
        'Language': df['programming_language'].astype(str),
        'Patch_Size': pd.to_numeric(df['patch_size'], errors='coerce').fillna(0).astype(int)
    })
    # Label mapping: CWE string -> integer class
    cwe_str = df['cwe_id'].astype(str)
    classes = sorted(cwe_str.unique())
    class_to_int = {c: i for i, c in enumerate(classes)}
    y = cwe_str.map(class_to_int).astype(int)
    # Save
    X.to_csv(X_OUT, index=False)
    pd.DataFrame({'label': y}).to_csv(Y_OUT, index=False)
    # Also save mapping for reference
    map_path = os.path.join(DATA_DIR, 'cwe_label_map.csv')
    pd.DataFrame({'cwe_id': classes, 'label': [class_to_int[c] for c in classes]}).to_csv(map_path, index=False)
    print(X_OUT)
    print(Y_OUT)
    print(map_path)

if __name__ == '__main__':
    main()