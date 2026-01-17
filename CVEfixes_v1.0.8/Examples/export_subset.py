import os
import sqlite3
import gzip
import csv
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'Data')
DB_PATH = os.path.join(DATA_DIR, 'CVEfixes.db')
SQL_PATH = os.path.join(DATA_DIR, 'CVEfixes_v1.0.8.sql')
SQL_GZ_PATH = os.path.join(DATA_DIR, 'CVEfixes_v1.0.8.sql.gz')
OUT_CSV = os.path.join(DATA_DIR, 'cve_dataset_subset.csv')

def ensure_db():
    create = False
    if not os.path.exists(DB_PATH):
        create = True
    con = sqlite3.connect(DB_PATH)
    con.execute('PRAGMA journal_mode=WAL')
    cur = con.cursor()
    if not create:
        try:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cve'")
            row = cur.fetchone()
            if not row:
                create = True
        except Exception:
            create = True
    if create:
        if os.path.exists(SQL_PATH):
            with open(SQL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                stmt = ''
                for line in f:
                    stmt += line
                    if line.strip().endswith(';'):
                        cur.executescript(stmt)
                        con.commit()
                        stmt = ''
                if stmt:
                    cur.executescript(stmt)
                    con.commit()
        elif os.path.exists(SQL_GZ_PATH):
            with gzip.open(SQL_GZ_PATH, 'rt', encoding='utf-8', errors='ignore') as f:
                stmt = ''
                for line in f:
                    stmt += line
                    if line.strip().endswith(';'):
                        cur.executescript(stmt)
                        con.commit()
                        stmt = ''
                if stmt:
                    cur.executescript(stmt)
                    con.commit()
    con.close()

def derive_project_name(repo_name, repo_url):
    if repo_name and repo_name.strip():
        return repo_name.strip()
    if not repo_url:
        return ''
    m = re.match(r'^https?://[^/]+/([^/]+/[^/]+)', repo_url)
    if m:
        return m.group(1)
    return repo_url

def build_query(top_cwe_limit=15, patch_threshold=5, restrict_top_cwe=True):
    top_clause = f"LIMIT {top_cwe_limit}" if restrict_top_cwe else ""
    q = f"""
    WITH recent_commits AS (
      SELECT hash, repo_url, committer_date
      FROM commits
      WHERE committer_date BETWEEN '2020-01-01' AND '2024-12-31'
    ),
    top_cwe AS (
      SELECT cc.cwe_id
      FROM cwe_classification cc
      JOIN fixes fx ON fx.cve_id = cc.cve_id
      JOIN recent_commits rc ON rc.hash = fx.hash
      GROUP BY cc.cwe_id
      ORDER BY COUNT(*) DESC
      {top_clause}
    ),
    method_rows AS (
      SELECT cv.cve_id, r.repo_name, rc.repo_url, fc.programming_language,
             CASE WHEN fc.new_path IS NOT NULL AND fc.new_path <> 'None' THEN fc.new_path
                  WHEN fc.old_path IS NOT NULL AND fc.old_path <> 'None' THEN fc.old_path
                  ELSE fc.filename END AS file_path,
             mc.name AS function_name, cc.cwe_id,
             COALESCE(fc.num_lines_added,0)+COALESCE(fc.num_lines_deleted,0) AS patch_size,
             rc.hash AS fix_commit_hash
      FROM fixes fx
      JOIN recent_commits rc ON rc.hash = fx.hash
      JOIN cve cv ON cv.cve_id = fx.cve_id
      JOIN cwe_classification cc ON cc.cve_id = cv.cve_id {('AND cc.cwe_id IN (SELECT cwe_id FROM top_cwe)') if restrict_top_cwe else ''}
      JOIN file_change fc ON fc.hash = rc.hash
      JOIN method_change mc ON mc.file_change_id = fc.file_change_id AND mc.before_change='True'
      LEFT JOIN repository r ON r.repo_url = rc.repo_url
    ),
    file_rows AS (
      SELECT cv.cve_id, r.repo_name, rc.repo_url, fc.programming_language,
             CASE WHEN fc.new_path IS NOT NULL AND fc.new_path <> 'None' THEN fc.new_path
                  WHEN fc.old_path IS NOT NULL AND fc.old_path <> 'None' THEN fc.old_path
                  ELSE fc.filename END AS file_path,
             NULL AS function_name, cc.cwe_id,
             COALESCE(fc.num_lines_added,0)+COALESCE(fc.num_lines_deleted,0) AS patch_size,
             rc.hash AS fix_commit_hash
      FROM fixes fx
      JOIN recent_commits rc ON rc.hash = fx.hash
      JOIN cve cv ON cv.cve_id = fx.cve_id
      JOIN cwe_classification cc ON cc.cve_id = cv.cve_id {('AND cc.cwe_id IN (SELECT cwe_id FROM top_cwe)') if restrict_top_cwe else ''}
      JOIN file_change fc ON fc.hash = rc.hash
      LEFT JOIN method_change mc ON mc.file_change_id = fc.file_change_id
      LEFT JOIN repository r ON r.repo_url = rc.repo_url
      WHERE mc.file_change_id IS NULL
    )
    SELECT cve_id, repo_name, repo_url, programming_language, file_path, function_name, cwe_id, patch_size, fix_commit_hash
    FROM (
      SELECT * FROM method_rows
      UNION ALL
      SELECT * FROM file_rows
    ) all_rows
    WHERE patch_size >= {patch_threshold}
    LIMIT 1000
    """
    return q

def export_csv():
    ensure_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    query = build_query(15, 5, True)
    rows = cur.execute(query).fetchall()
    if len(rows) < 500:
        query = build_query(20, 5, True)
        rows = cur.execute(query).fetchall()
    if len(rows) < 500:
        query = build_query(20, 3, False)
        rows = cur.execute(query).fetchall()
    headers = ['cve_id','project_name','programming_language','file_path','function_name','cwe_id','patch_size','fix_commit_hash']
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            project_name = derive_project_name(r['repo_name'], r['repo_url'])
            w.writerow([
                r['cve_id'],
                project_name,
                r['programming_language'] or '',
                r['file_path'] or '',
                r['function_name'] or '',
                r['cwe_id'] or '',
                int(r['patch_size']) if r['patch_size'] is not None else 0,
                r['fix_commit_hash'] or ''
            ])
    o = csv.writer(open(os.devnull, 'w'))
    print(','.join(headers))
    for r in rows:
        project_name = derive_project_name(r['repo_name'], r['repo_url'])
        print(','.join([
            str(r['cve_id'] or ''),
            str(project_name or ''),
            str(r['programming_language'] or ''),
            str(r['file_path'] or ''),
            str(r['function_name'] or ''),
            str(r['cwe_id'] or ''),
            str(int(r['patch_size']) if r['patch_size'] is not None else 0),
            str(r['fix_commit_hash'] or '')
        ]))
    con.close()
    return len(rows)

if __name__ == '__main__':
    count = export_csv()
    print(OUT_CSV)
    print(count)