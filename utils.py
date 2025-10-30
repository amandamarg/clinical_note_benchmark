from dotenv import load_dotenv
import pandas as pd
import os
import requester
from tqdm import tqdm
from rouge import Rouge
import json
import re    
from glob import glob


def get_standard_path(idx, standard_dir='standards'):
    if f'{str(idx)}.txt' in os.listdir(standard_dir):
        path = os.readlink(os.path.join(standard_dir, f'{str(idx)}.txt'))
        return path
    else:
        raise FileNotFoundError(f'Standard note for idx {str(idx)} not found in {standard_dir}')

def get_gen_paths(results_dir_path='results', idxs='all', models='all', prompts='all'):
    if idxs == 'all':
        idxs = r'\d+'
    else:
        idxs = ('|').join(list(map(str, idxs)))
    if models == 'all':
        models = r'.+'
    elif isinstance(models, list):
        models = ('|').join(models)
    if prompts == 'all':
        prompts = r'g\d+'
    elif isinstance(prompts, list):
        prompts = ('|').join(prompts)
    pattern = re.compile(rf'{results_dir_path}/({idxs})/({models})/({prompts})/(\d+\.\d+)/gen_note.txt')
    glob_paths = glob(f'{results_dir_path}/**/gen_note.txt', recursive=True)
    filtered_paths = list(filter(pattern.match, glob_paths))
    return filtered_paths    

def parse_path(path):
    pattern = re.compile(rf'(.*)/(\d+)/(.+)/(.+)/(\d+\.\d+)/gen_note.txt')
    match = pattern.match(path)
    assert match
    return {
        'full_path': path,
        'root_dir': match[1],
        'idx': int(match[2]),
        'model': match[3],
        'prompt': match[4],
        'timestamp': match[5]
    }

def read(path):
    with open(path, 'r') as file:
        content = file.read()
    return content


def get_metadata_df(paths):
    notes = list(map(read, paths))
    metadata = list(map(parse_path, paths))
    df = pd.DataFrame.from_records(metadata)
    df['note'] = notes
    return df


def write_reports(path, df):
    if os.path.exists(path):
        existing_report = pd.read_json(path)
        pd.concat((existing_report, df)).reset_index(drop=True).to_json(path, orient='records', indent=4)
    else:
        df.to_json(path, orient='records', indent=4)
