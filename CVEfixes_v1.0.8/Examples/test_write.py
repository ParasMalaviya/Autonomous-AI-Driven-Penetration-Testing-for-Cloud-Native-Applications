import os
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', 'cve_dataset_subset.csv')
with open(p, 'w', encoding='utf-8') as f:
    f.write('a,b\n1,2\n')
print(p)