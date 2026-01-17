import os
import sqlite3
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'Data')
DB_PATH = os.path.join(DATA_DIR, 'CVEfixes.db')
OUT_PATH = os.path.join(DATA_DIR, 'cvefixes_subset.csv')

def query(conn):
    q = """
    WITH fc_agg AS (
      SELECT fc.hash AS hash,
             COUNT(*) AS num_files_changed,
             SUM(COALESCE(fc.num_lines_added,0)) AS lines_added,
             SUM(COALESCE(fc.num_lines_deleted,0)) AS lines_deleted
      FROM file_change fc
      GROUP BY fc.hash
    ), mc_agg AS (
      SELECT fc.hash AS hash,
             COUNT(mc.method_change_id) AS num_functions_changed
      FROM file_change fc
      LEFT JOIN method_change mc ON mc.file_change_id = fc.file_change_id
      GROUP BY fc.hash
    ), lang_pick AS (
      SELECT hash, programming_language
      FROM (
        SELECT fc.hash AS hash,
               fc.programming_language AS programming_language,
               ROW_NUMBER() OVER (PARTITION BY fc.hash ORDER BY fc.programming_language) AS rn
        FROM file_change fc
      ) t
      WHERE rn=1
    ), cwe_pick AS (
      SELECT cc.cve_id AS cve_id,
             MIN(cc.cwe_id) AS cwe_id
      FROM cwe_classification cc
      GROUP BY cc.cve_id
    )
    SELECT fx.cve_id AS cve_id,
           cp.cwe_id AS cwe_id,
           c.hash AS hash,
           c.msg AS commit_message,
           c.committer_date AS committer_date,
           r.date_created AS repo_created,
           COALESCE(r.stars_count,0) AS stars,
           COALESCE(r.forks_count,0) AS forks,
           COALESCE(fa.num_files_changed,0) AS num_files_changed,
           COALESCE(ma.num_functions_changed,0) AS num_functions_changed,
           COALESCE(fa.lines_added,0) AS lines_added,
           COALESCE(fa.lines_deleted,0) AS lines_deleted,
           lp.programming_language AS language,
           r.repo_url AS repo_url,
           c.author AS author
    FROM fixes fx
    JOIN commits c ON c.hash = fx.hash
    LEFT JOIN repository r ON r.repo_url = c.repo_url
    LEFT JOIN fc_agg fa ON fa.hash = c.hash
    LEFT JOIN mc_agg ma ON ma.hash = c.hash
    LEFT JOIN lang_pick lp ON lp.hash = c.hash
    LEFT JOIN cwe_pick cp ON cp.cve_id = fx.cve_id
    WHERE c.committer_date BETWEEN '2020-01-01' AND '2024-12-31'
    """
    df = pd.read_sql_query(q, conn)
    return df

def compute_extra(df):
    df['commit_message_length'] = df['commit_message'].fillna('').astype(str).str.len()
    def repo_age_days(row):
        try:
            return (pd.to_datetime(row['committer_date']) - pd.to_datetime(row['repo_created'])).days
        except Exception:
            return np.nan
    df['repository_age'] = df.apply(repo_age_days, axis=1).fillna(0).astype(int)
    df['has_cwe_description'] = (df['cwe_id'].fillna('') != '').astype(int)
    df = df.drop(columns=['commit_message', 'committer_date', 'repo_created', 'repo_url', 'author'])
    lang_dummies = pd.get_dummies(df['language'].fillna('Unknown'), prefix='language')
    df = pd.concat([df.drop(columns=['language']), lang_dummies], axis=1)
    return df

def main():
    con = sqlite3.connect(DB_PATH)
    df = query(con)
    con.close()
    df = compute_extra(df)
    df = df.dropna(subset=['cwe_id'])
    df = df.sample(frac=0.2, random_state=42) if len(df) > 0 else df
    df.to_csv(OUT_PATH, index=False)
    print(OUT_PATH)

if __name__ == '__main__':
    main()