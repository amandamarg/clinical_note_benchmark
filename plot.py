import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import re

MODEL_NAME = 'ozwell'
PROMPT_NAME = 'g1'
FILE_NAME = 'eval_report'

if __name__ == '__main__':
    
    path = os.path.join('expiriments', MODEL_NAME, PROMPT_NAME)
    data = pd.read_json(os.path.join(path, FILE_NAME + '.json'))
    rouge_dfs = []
    for rt in ['rouge-1', 'rouge-2', 'rouge-l']:
        rouge_dfs.append(pd.DataFrame.from_records(data[rt].values).rename(columns={k:f'{rt}-{k}' for k in ['f', 'p', 'r']}))
    
    df = pd.concat(rouge_dfs, axis=1)
    df = pd.concat((df, data.get(['idx', 'standard_note_path'])), axis=1)
    df['standard_note_path'] = df['standard_note_path'].map(lambda x: x.split('/')[-1])
    
    for rt in ['rouge-1', 'rouge-2', 'rouge-l']:
        fig, ax = plt.subplots(3, 1, figsize=(15, 10), layout='constrained')
        for i,c in enumerate(['f', 'p', 'r']):
            sns.scatterplot(data=df, x='idx', y=f'{rt}-{c}', hue='standard_note_path', ax=ax[i])
            ax[i].legend(loc='lower right', bbox_to_anchor=(1.1, 0))
        title = f'{rt} scores'
        if re.search(r'rouge_avgs', FILE_NAME):
            title = 'avg ' + title
        fig.suptitle(title, fontsize=16)
        fig.savefig(os.path.join(path, 'plots', title.replace(' ', '_') + '.png'))

