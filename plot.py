import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_rouge_report, search_file_paths, melt_rouge_scores, get_standard_path, parse_path, get_most_recent_timestamp
import pandas as pd
import os


RESULTS_DIR = 'results'
IDXS = 'all' # or 'all'
MODEL_NAMES = ['ozwell'] # list of model names to include in the plot
PROMPT_NAMES = ['g1', 'g2'] # list of prompt names to include in the plot
STANDARDS_AGGR = 'use_set' # 'avg' to include averaged standards for each for each idx/model/prompt/timestamp, 'use_set' to uses whatever standards are currently set in 'standards' directory for each idx, or None to plot all standards
TIMESTAMP_AGGR = 'avg' # 'avg' to average across timestamps for each idx/model/prompt, 'most_recent' to use the most recent timestamp for each idx/model/prompt, or None to plot all timestamps (occurs after standards aggregation)
X_CATEGORY = 'idx' # 'model', 'prompt', or 'idx' (or 'timestamp' or 'standard_note_path' if not aggregated)
COLOR_CATEGORY = 'prompt' # None, 'model', 'prompt', or 'idx' (or 'timestamp' or 'standard_note_path' if not aggregated)
SAVE_PATH = 'plots/avg_rouge_plot.png'
ROUGE_TYPES = ['rouge-1', 'rouge-2', 'rouge-l']


def aggr_standards(df, method):
    if method == 'avg':
        df = df.groupby([c for c in df.columns if c not in ['standard_note_path', 'score']])['score'].mean().reset_index()
    elif method == 'use_set':
        df = df[df['standard_note_path'] == df['idx'].map(get_standard_path)].reset_index(drop=True)
    return df

def aggr_timestamps(df, method):
    if method == 'avg':
        df = df.groupby([c for c in df.columns if c not in ['timestamp', 'score']])['score'].mean().reset_index()
    elif method == 'most_recent':
        timestamp_idx = df.apply(lambda row: get_most_recent_timestamp(row['root_dir'], row['model'], row['prompt'], row['idx']), axis=1)
        df = df[df['timestamp'] == timestamp_idx].reset_index(drop=True)
    return df

def plotter(df, x_category, color_category, save_path, col_name, row_name):
    g = sns.catplot(data=df, x=x_category, y='score', hue=color_category, col=col_name, row=row_name, dodge=True, legend='full', aspect=1.5)
    g.savefig(save_path)


if __name__ == '__main__':
    if STANDARDS_AGGR:
        assert X_CATEGORY != 'standard_note_path', "Cannot use 'standard_note_path' as X_CATEGORY when STANDARDS_AGGR is set"
        assert COLOR_CATEGORY != 'standard_note_path', "Cannot use 'standard_note_path' as COLOR_CATEGORY when STANDARDS_AGGR is set"
    if TIMESTAMP_AGGR:
        assert X_CATEGORY != 'timestamp', "Cannot use 'timestamp' as X_CATEGORY when TIMESTAMP_AGGR is set"
        assert COLOR_CATEGORY != 'timestamp', "Cannot use 'timestamp' as COLOR_CATEGORY when TIMESTAMP_AGGR is set"

    eval_df = []
    for rouge_type in ROUGE_TYPES:
        eval_report_paths = search_file_paths(filename=f'{rouge_type}.json', results_dir_path=RESULTS_DIR, idxs=IDXS, models=MODEL_NAMES, prompts=PROMPT_NAMES)
        eval_df.extend(list(map(get_rouge_report, eval_report_paths)))
    eval_df = pd.concat(eval_df).reset_index(drop=True)
    if STANDARDS_AGGR:
        eval_df = aggr_standards(eval_df, STANDARDS_AGGR)
    if TIMESTAMP_AGGR:
        eval_df = aggr_timestamps(eval_df, TIMESTAMP_AGGR)
    plotter(eval_df, X_CATEGORY, COLOR_CATEGORY, SAVE_PATH, 'rouge_type', 'metric_type')

