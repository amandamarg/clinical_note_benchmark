import matplotlib.pyplot as plt
import json
import pandas as pd
import os
import requester
import seaborn as sns

def plot_results(rouge_scores, name):
    fig, axs = plt.subplots(len(rouge_scores), 1, figsize=(10, 10))
    for i, (k,v) in enumerate(rouge_scores.items()):
        axs[i].set_title(k)
        for kk,vv in v.items():
            axs[i].scatter(vv['labels'], vv['data'], label=kk)
    fig.legend()
    fig.suptitle(name)
    fig.savefig(f'expiriments/plots/{name}.png')

def plot_rouge_components(df, label_col, rouge_type, save_dir='./'):
    fig, axs = plt.subplot_mosaic([['f', 'p'],['r', 'all']], figsize=(20,12))
    score_components = {'f': 'F1', 'p': 'Precision', 'r': 'Recall'}
    color_map = {'f': 'blue', 'p': 'green', 'r': 'red'}
    for sc in ['f', 'p', 'r']:
        scores = df['rouge'].map(lambda x: x[rouge_type][sc]).values
        labels = df[label_col].values
        axs[sc].scatter(labels, scores, c=color_map[sc])
        axs[sc].set_title(f'{score_components[sc]} Scores')
        axs[sc].set_xlabel(label_col)
        axs[sc].set_ylabel(f'Scores')
        axs['all'].scatter(labels, scores, label=sc, c=color_map[sc])
    axs['all'].set_title(f'All Components')
    axs['all'].set_xlabel(label_col)
    axs['all'].set_ylabel(f'Scores')
    axs['all'].legend()
    fig.suptitle(f'{rouge_type} Scores by {label_col}')
    fig.savefig(os.path.join(save_dir, f'{rouge_type}-{label_col}-.png'))
    return fig

def plot_groups(df, group_name, x_label, y_label, mosaic_labels, color_map=None, save_dir='./'):
    fig, axs = plt.subplot_mosaic(mosaic_labels, figsize=(20,12))
    for k,v in df.groupby(group_name):
        axs[k].scatter(v[x_label].values, v[y_label].values, c=color_map[k] if color_map else None)
        axs[k].set_xlabel(x_label)
        axs[k].set_ylabel(y_label)
        axs[k].set_title(k)
        axs['all'].scatter(v[x_label].values, v[y_label].values, c=color_map[k] if color_map else None, label=k)
    axs['all'].legend(title=group_name)
    axs['all'].set_title(f'All {group_name}')
    axs['all'].set_xlabel(x_label)
    axs['all'].set_ylabel(y_label)
    fig.suptitle(f'{y_label} by {x_label} and {group_name}')
    fig.savefig(os.path.join(save_dir, f'{y_label}-by-{x_label}-and-{group_name}.png'))
    return fig



def plot(fig_num, df, x_label, rouge_type, score_component, group_label=None, save_dir='./'):
    scores = df['rouge'].map(lambda x: x[rouge_type][score_component]).values
    score_components = {'f': 'F1', 'p': 'Precision', 'r': 'Recall'}
    plt.figure(fig_num, figsize=(10,6))
    if group_label:
        plt.scatter(df[x_label].values, scores, label=df[group_label].values)
        plt.legend(title=group_label)
    else:
        plt.scatter(df[x_label], scores)
    plt.xlabel(x_label)
    plt.ylabel(f'{rouge_type} {score_components[score_component]} Scores')
    plt.title(f'{rouge_type} {score_components[score_component]} Scores by {x_label}')
    plt.savefig(os.path.join(save_dir, f'{rouge_type}-{score_component}-by-{x_label}.png'))


if __name__ == '__main__':
    # with open('expiriments/rouge_avgs.json', 'r') as file:
    #     avgs = json.load(file)

    # scores = {}
    # for x in avgs:
    #     if x['standard'] not in scores.keys():
    #         scores[x['standard']] = {}
    #     for k,v in x['results'].items():
    #         if k not in scores[x['standard']].keys():
    #             scores[x['standard']][k] = {}
    #         for kk,vv in v.items():
    #             if kk not in scores[x['standard']][k].keys():
    #                 scores[x['standard']][k][kk] = {'labels':[], 'data':[]}
    #             scores[x['standard']][k][kk]['labels'].append(x['gen_note'])
    #             scores[x['standard']][k][kk]['data'].append(vv)

    # for k,v in scores.items():
    #     plot_results(v, k)
    eval_report = pd.read_json('expiriments/ozwell/g2/eval_report1.json')
    # x = pd.concat((eval_report.get(['idx']), pd.DataFrame.from_records(eval_report['rouge-l'].values)), axis=1)
    rouge_dfs = []
    for rt in ['rouge-1', 'rouge-2', 'rouge-l']:
        rouge_dfs.append(pd.DataFrame.from_records(eval_report[rt].values).rename(columns={k:f'{rt}-{k}' for k in ['f', 'p', 'r']}))
    
    df = pd.concat(rouge_dfs, axis=1)
    df = pd.concat((df, eval_report.get(['idx', 'gen_note_path', 'standard_note_path'])), axis=1)
    # df.plot.scatter(x='idx', y='rouge-l-f', figsize=(10,6), c='blue', label='F1', ylabel='Scores', title='Rouge-L F1 Scores by idx')
    df['standard_note_path'] = df['standard_note_path'].map(lambda x: x.split('/')[-1])
    # ax = df.groupby('standard_note_path').plot(kind='scatter', x='idx', y='rouge-l-f', figsize=(10,6), ylabel='Scores', title='Rouge-L F1 Scores by idx')
    # df.groupby('standard_note_path').plot(kind='scatter',  x='idx', y='rouge-l-f', legend=True)
    sns.scatterplot(data=df, x='idx', y='rouge-l-f', hue='standard_note_path')
    plt.show()
    # x = pd.concat((pd.DataFrame.from_records(eval_report['rouge-l'].values), , eval_report.get(['idx'])), axis=1)
    # ax = x.plot.scatter(x='idx', y='p', figsize=(10,6), c='blue', label='Precision', ylabel='Scores', title='Rouge-L Scores by idx')
    # x.plot.scatter(ax=ax, x='idx', y='r', figsize=(10,6), c='red', label='Recall', ylabel='Scores', title='Rouge-L Scores by idx')
    # x.plot.scatter(ax=ax, x='idx', y='f', figsize=(10,6), c='green', label='F1', ylabel='Scores', title='Rouge-L Scores by idx')
    # plt.savefig('expiriments/ozwell/g2/plots/rouge-l-by-idx.png')
    # ax = x.plot()
    # plt.show()
    # .plot.scatter(x='idx', y='rouge-l', figsize=(10,6))
    # plot_groups(eval_report['rouge-l'], 'Score Components', 'idx', 'rouge-l scores', [['f', 'p'], ['r', 'all']], color_map={'f': 'blue', 'p': 'green', 'r': 'red'}, save_dir='expiriments/ozwell/g2/plots')
    # for rt in rouge_types:
    #     data = {}
    #     for sc in score_components:
    #         data[sc] = eval_report['rouge'].map(lambda x: x[rt][sc])

    # rouge_idx_avgs = pd.read_json('expiriments/ozwell/g2/rouge_idx_avgs.json')
    # plot(rouge_idx_avgs, 'idx', 'rouge-1', 'p', save_dir='expiriments/ozwell/g2/plots')