import pandas as pd
import os

cols = ['id','label','statement','subject','speaker','job',
        'state','party','barely_true','false_count',
        'half_true','mostly_true','pants_fire','context']

for split in ['train', 'valid', 'test']:
    tsv_file = f'{split}.tsv'
    csv_file = f'{split}_batch.csv'

    if os.path.exists(tsv_file):
        df = pd.read_csv(tsv_file, sep='\t', header=None, names=cols)
        df[['statement', 'label']].to_csv(csv_file, index=False)
        print(f'Saved {csv_file} with {len(df)} rows')
    else:
        print(f'File not found: {tsv_file}')    