from dotenv import load_dotenv
import pandas as pd
import os
import requester
from evaluate import Evaluator
from generate import generate_loop, generate_n
from utils import get_gen_note_paths, read_gen_notes, set_standard, write_file
import json
import re
from utils import match_filenames

N = 3
STANDARD_VERSION = '1'

if __name__ == '__main__':
    load_dotenv()
    df = pd.read_json('augmented-clinical-notes/augmented_notes_30K.jsonl', lines=True)
    samples = ['224', '431', '562', '619', '958', '1380', '1716', '1834', '2021', '3026', '3058', '3093', '3293', '3931', '4129']
    df_samples = df[df['idx'].astype('string').isin(samples)]
    
    #Ozwell
    req = requester.OzwellRequester("g2", os.getenv("OZWELL_SECRET_KEY"))
    gen_name = req.model_name + '_' + req.prompt_name
    for s in samples:
        if STANDARD_VERSION != 'ref':
            set_standard(s, os.path.join('ozwell', 'g2', str(s), f'gen_note{str(STANDARD_VERSION) if str(STANDARD_VERSION) != '0' else ''}.txt'))
        else:
            set_standard(s, 'ref')
    generate_n(N, df_samples, req)
    eval = Evaluator()
    req.set_prompt('s1')
    eval_name = req.model_name + '-' + req.prompt_name
    avgs = {'global':{}, 'indiv':{}}
    eval_data = []
    for i in range(N):
        gen_note_paths = get_gen_note_paths('ozwell/g2', samples, i)
        gen_notes = read_gen_notes(gen_note_paths)
        standards = eval.get_standards(gen_notes)
        ai_responses = eval.ai_eval(gen_notes, req)
        rouge_scores = eval.get_rouge(gen_notes, False)
        eval_data.extend(list(zip(gen_notes['idx'].values, gen_notes['path'].values, standards['path'].values, ai_responses, rouge_scores)))
        avgs['global'][f'v{i}'] = eval.get_rouge(gen_notes, True)
    eval_df = pd.DataFrame(eval_data, columns=['idx', 'gen_note_path', 'standard_note_path', eval_name, 'rouge'])
    dest_path = os.path.join('expiriments', gen_name)
    if os.path.exists(os.path.join(dest_path, f'eval_report.json')):
        existing_report = pd.read_json(dest_path)
        pd.concat((existing_report, eval_df)).reset_index(drop=True).to_json(dest_path)
    else:
        eval_df.to_json(dest_path)
    for s in samples:
        gen_note_paths = get_gen_note_paths(gen_name, s)
        gen_notes = read_gen_notes(gen_note_paths)
        avgs['indiv'][s] = eval.get_rouge(gen_notes, True)

    matches = match_filenames(dest_path, 'rouge_avgs', 'json')
    if len(matches) > 0:
        version = max([0 if (m[1] == '') else int(m[1]) for m in matches])
        filename = 'rouge_avgs.json' if version == 0 else f'rouge_avgs{version}.json'
        with open(os.path.join(dest_path, filename), 'r') as file:
            existing = json.load(file)
        existing[STANDARD_VERSION] = avgs
        with open(os.path.join(dest_path, f'rouge_avgs{version+1}'), 'w') as file:
            json.dump(existing, file)
    else:
        with open(os.path.join(dest_path, 'rouge_avgs.json'), 'w') as file:
            json.dump({STANDARD_VERSION: avgs}, file)



