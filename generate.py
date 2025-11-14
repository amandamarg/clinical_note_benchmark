import os
from tqdm import tqdm
import time
import requester
import pandas as pd
import json

'''
Run this script to generate clinical notes using a specified model and prompt.
'''
GENERATE_MODEL_NAME = 'ozwell'
GENERATE_PROMPT_NAME = 'g2'
# IDXS = [224, 431, 562, 619, 958, 1380, 1716, 1834, 2021, 3026, 3058, 3093, 3293, 3931, 4129] # or 'all'
IDXS = [155216]
def init_dirs(df, root='./'):
    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=50):
        if str(idx) not in os.listdir(os.path.join(root, 'results')):
            os.makedirs(os.path.join(root, 'results', str(idx)))
        for k,v in row.to_dict().items():
            if k == 'summary':
                if not os.path.exists(os.path.join(root, 'results', str(idx), f'{k}.json')):
                    with open(os.path.join(root, 'results', str(idx), f'{k}.json'), 'w') as f:
                        json.dump(v, f)
            else:
                if not os.path.exists(os.path.join(root, 'results', str(idx), f'{k}.txt')):
                    with open(os.path.join(root, 'results', str(idx), f'{k}.txt'), 'w') as f:
                        f.write(str(v))

def generate(df, generator, root='./'):
    df = df.copy()
    if 'idx' in df.columns:
        df = df.set_index('idx')
    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=50):
        timestamp = str(time.time())
        path = os.path.join(root, 'results', str(idx), generator.model_name, generator.prompt_name, timestamp)
        if not os.path.exists(path):
            os.makedirs(path)
        gen_note = generator.send(row['conversation'])
        with open(os.path.join(path, "gen_note.txt"), 'w') as f:
            f.write(gen_note)

if __name__ == '__main__':
    if GENERATE_MODEL_NAME == 'ozwell':
        req_gen = requester.OzwellRequester(GENERATE_PROMPT_NAME)
    else:
        req_gen = requester.OllamaRequester(GENERATE_MODEL_NAME, GENERATE_PROMPT_NAME)
    
    df = pd.read_json('augmented-clinical-notes/augmented_notes_30K.jsonl', lines=True)
    if IDXS != 'all':
        df = df[df['idx'].isin(IDXS)].reset_index(drop=True)
    df = df.set_index('idx')
    init_dirs(df, root='./')
    generate(df, req_gen)
