import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import re

MODEL_NAME = 'ozwell'
PROMPT_NAME = 'g2'
FILE_NAME = 'eval_report'

def plot(path, df, x_col_name='idx', grouping_col_name='standard_note_path', grouping_legend_title='standard note'):
    assert grouping_col_name in df.columns and x_col_name in df.columns
    df[grouping_col_name] = df[grouping_col_name].map(lambda x: x.split('/')[-1])
    for rt in ['rouge-1', 'rouge-2', 'rouge-l']:
            fig, ax = plt.subplots(3, 1, figsize=(15, 10), layout='constrained')
            for i,c in enumerate(['f', 'p', 'r']):
                sns.scatterplot(data=df, x=x_col_name, y=f'{rt}-{c}', hue=grouping_col_name, ax=ax[i])
                ax[i].legend(loc='lower right', bbox_to_anchor=(1.1, 0), title=grouping_legend_title)
            title = f'{rt} scores by {x_col_name} and {grouping_legend_title}'
            if re.search(r'rouge_avgs', FILE_NAME):
                title = 'avg ' + title
            fig.suptitle(title, fontsize=16)
            fig.savefig(os.path.join(path, 'plots',  title.replace(' ', '_') + f'.png'))

def reformated_df(df, rouge_types=['rouge-1', 'rouge-2', 'rouge-l']):
    rouge_dfs = []
    for rt in rouge_types:
        rouge_dfs.append(pd.DataFrame.from_records(df[rt].values).rename(columns={k:f'{rt}-{k}' for k in ['f', 'p', 'r']}))
    r_df = pd.concat(rouge_dfs, axis=1)
    cols = [c for c in df.columns if c not in rouge_types]
    return pd.concat((r_df, df.get(cols)), axis=1)
    

if __name__ == '__main__':
    
    path = os.path.join('expiriments', MODEL_NAME, PROMPT_NAME)
    df = pd.read_json(os.path.join(path, FILE_NAME + '.json'))
    if re.search(r'eval_report', FILE_NAME):
        # exclude comparisons between identical notes
        df = df[df['standard_note_path'] != df['gen_note_path']].reset_index(drop=True).copy()
    df = reformated_df(df)
    plot(path, df)
    if 'gen_note_path' in df.columns:
        plot(path, df, grouping_col_name='gen_note_path', grouping_legend_title='generated note')
