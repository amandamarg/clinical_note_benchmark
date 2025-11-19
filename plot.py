import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_rouge_report, search_file_paths, melt_rouge_scores, get_standard_path, parse_path
import pandas as pd
import os

# RESULTS_DIR = 'results'
# IDXS = [224, 431, 562, 619, 958, 1380, 1716, 1834, 2021, 3026, 3058, 3093, 3293, 3931, 4129] # or 'all'
# MODEL_NAMES = ['ozwell'] # list of model names to include in the plot
# PROMPT_NAMES = ['g1'] # list of prompt names to include in the plot
# STANDARDS_AGGR = 'avg' # 'avg' to include averaged standards for each for each idx/model/prompt/timestamp, 'use_set' to uses whatever standards are currently set in 'standards' directory for each idx, or None to plot all standards
# TIMESTAMP_AGGR = 'avg' # 'avg' to average across timestamps for each idx/model/prompt, 'most_recent' to use the most recent timestamp for each idx/model/prompt, or None to plot all timestamps (occurs after standards aggregation)
# X_CATEGORY = 'idx' # 'model', 'prompt', or 'idx' (or 'timestamp' or 'standard_note_path' if not aggregated)
# COLOR_CATEGORY = None # None, 'model', 'prompt', or 'idx' (or 'timestamp' or 'standard_note_path' if not aggregated)
# SAVE_PATH = 'plots/rouge_plot.png'


# if __name__ == '__main__':
#     if STANDARDS_AGGR:
#         assert X_CATEGORY != 'standard_note_path', "Cannot use 'standard_note_path' as X_CATEGORY when STANDARDS_AGGR is set"
#         assert COLOR_CATEGORY != 'standard_note_path', "Cannot use 'standard_note_path' as COLOR_CATEGORY when STANDARDS_AGGR is set"
#     if TIMESTAMP_AGGR:
#         assert X_CATEGORY != 'timestamp', "Cannot use 'timestamp' as X_CATEGORY when TIMESTAMP_AGGR is set"
#         assert COLOR_CATEGORY != 'timestamp', "Cannot use 'timestamp' as COLOR_CATEGORY when TIMESTAMP_AGGR is set"

#     eval_report_paths = search_file_paths(filename='', results_dir_path=RESULTS_DIR, idxs=IDXS, models=MODEL_NAMES, prompts=PROMPT_NAMES)
   
#     eval_df = []
#     for path in eval_report_paths:
#         eval_df.append(pd.read_json(path))
#     eval_df = melt_rouge_scores(pd.concat(eval_df)).reset_index(drop=True)
#     if STANDARDS_AGGR == 'avg':
#         eval_df = eval_df.groupby(['idx', 'gen_note_path', 'rouge_type', 'metric'])['value'].mean()
#         eval_df = eval_df.to_frame().reset_index()
#     elif STANDARDS_AGGR == 'use_set':
#         eval_df = eval_df[eval_df['standard_note_path'] == eval_df['idx'].map(get_standard_path)].reset_index(drop=True)
#     eval_df = pd.concat((eval_df, pd.json_normalize(eval_df['gen_note_path'].map(parse_path)).drop(columns=['full_path', 'root_dir', 'idx'])), axis=1)
#     if TIMESTAMP_AGGR == 'avg':
#         eval_df = eval_df.groupby(['idx', 'model', 'prompt', 'rouge_type', 'metric'])['value'].mean()
#         eval_df = eval_df.to_frame().reset_index()
#     elif TIMESTAMP_AGGR == 'most_recent':
#         most_recent_timestamps = eval_df.groupby(['idx', 'model', 'prompt'])['timestamp'].max().to_frame().reset_index()
#         most_recent_timestamps = pd.MultiIndex.from_frame(most_recent_timestamps)
#         eval_df.set_index(['idx', 'model', 'prompt', 'timestamp'], inplace=True)
#         eval_df = eval_df.loc[most_recent_timestamps].reset_index()
    
#     g = sns.catplot(data=eval_df, x=X_CATEGORY, y='value', hue=COLOR_CATEGORY, col='rouge_type', row='metric', dodge=True, legend='full')
#     g.savefig(SAVE_PATH)


RESULTS_DIR = 'results'
IDXS = [224, 431, 562, 619, 958, 1380, 1716, 1834, 2021, 3026, 3058, 3093, 3293, 3931, 4129] # or 'all'
MODEL_NAMES = ['ozwell'] # list of model names to include in the plot
PROMPT_NAMES = ['g1'] # list of prompt names to include in the plot
STANDARDS_AGGR = 'avg' # 'avg' to include averaged standards for each for each idx/model/prompt/timestamp, 'use_set' to uses whatever standards are currently set in 'standards' directory for each idx, or None to plot all standards
TIMESTAMP_AGGR = 'avg' # 'avg' to average across timestamps for each idx/model/prompt, 'most_recent' to use the most recent timestamp for each idx/model/prompt, or None to plot all timestamps (occurs after standards aggregation)
X_CATEGORY = 'idx' # 'model', 'prompt', or 'idx' (or 'timestamp' or 'standard_note_path' if not aggregated)
COLOR_CATEGORY = None # None, 'model', 'prompt', or 'idx' (or 'timestamp' or 'standard_note_path' if not aggregated)
SAVE_PATH = 'plots/rouge_plot.png'
ROUGE_TYPES = ['rouge-1', 'rouge-2', 'rouge-l']

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
    if STANDARDS_AGGR == 'avg':
        eval_df = eval_df.groupby([c for c in eval_df.columns if c not in ['rouge_type', 'metric_type', 'score']])['score'].mean().reset_index()
    elif STANDARDS_AGGR == 'use_set':
        eval_df = eval_df[eval_df['standard_note_path'] == eval_df['idx'].map(get_standard_path)].reset_index(drop=True)
    if TIMESTAMP_AGGR == 'avg':
        eval_df = eval_df.groupby([c for c in eval_df.columns if c not in ['timestamp', 'rouge_type', 'metric_type', 'score']])['score'].mean().reset_index()
    elif TIMESTAMP_AGGR == 'most_recent':
        most_recent_timestamps = eval_df.groupby([c for c in eval_df.columns if c not in ['timestamp', 'rouge_type', 'metric_type', 'score']])['timestamp'].max().reset_index()
        most_recent_timestamps = pd.MultiIndex.from_frame(most_recent_timestamps)
        eval_df.set_index(['idx', 'model', 'prompt', 'timestamp'], inplace=True)
        eval_df = eval_df.loc[most_recent_timestamps].reset_index()
    
    g = sns.catplot(data=eval_df, x=X_CATEGORY, y='value', hue=COLOR_CATEGORY, col='rouge_type', row='metric', dodge=True, legend='full')
    g.savefig(SAVE_PATH)
