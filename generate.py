import os
from tqdm import tqdm
import time
import requester
import pandas as pd

GENERATE_MODEL_NAME = 'ozwell'
GENERATE_PROMPT_NAME = 'g1'
IDXS = [224, 431, 562, 619, 958, 1380, 1716, 1834, 2021, 3026, 3058, 3093, 3293, 3931, 4129] # or 'all'

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
    if IDXS is not 'all':
        df = df[df['idx'].isin(IDXS)].reset_index(drop=True)
    generate(df, req_gen)
