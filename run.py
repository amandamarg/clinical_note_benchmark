from dotenv import load_dotenv
import pandas as pd
import os
import requester
from evaluate import Evaluator
from generate import generate_loop, generate_n
from utils import get_gen_note_paths, read_gen_notes, set_standard, write_file
import json
import re
from utils import match_filenames, extend_or_create

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
    global_avgs = []
    idx_avgs = []
    eval_data = []
    for i in range(N):
        gen_note_paths = get_gen_note_paths('ozwell/g2', samples, i)
        gen_notes = read_gen_notes(gen_note_paths)
        standards = eval.get_standards(gen_notes)
        # ai_responses = eval.ai_eval(gen_notes, req)
        rouge_scores = eval.get_rouge(gen_notes, False)
        eval_data.extend(list(zip(gen_notes['idx'].values, gen_notes['path'].values, standards['path'].values, rouge_scores)))
        # eval_data.extend(list(zip(gen_notes['idx'].values, gen_notes['path'].values, standards['path'].values, ai_responses, rouge_scores)))
        global_avgs.append((STANDARD_VERSION if STANDARD_VERSION == 'ref' else f'v{STANDARD_VERSION}', f'v{i}', eval.get_rouge(gen_notes, True)))
    # eval_df = pd.DataFrame(eval_data, columns=['idx', 'gen_note_path', 'standard_note_path', eval_name, 'rouge'])
    eval_df = pd.DataFrame(eval_data, columns=['idx', 'gen_note_path', 'standard_note_path', 'rouge'])
    print(global_avgs)
    # extend_or_create(os.path.join('expiriments', gen_name, f'eval_report.json'), eval_df)

    global_avgs_df = pd.DataFrame(global_avgs, columns=['standard', 'version_avged', 'results'])
    # extend_or_create(os.path.join('expiriments', gen_name, f'rouge_global_avgs.json'), global_avgs_df)

    # for s in samples:
    #     gen_note_paths = get_gen_note_paths(gen_name, s)
    #     gen_notes = read_gen_notes(gen_note_paths)
    #     idx_avgs.append((STANDARD_VERSION, s, eval.get_rouge(gen_notes, True)))
    # idx_avgs_df = pd.DataFrame(idx_avgs, columns=['standard', 'idx', 'results'])
    # extend_or_create(os.path.join('expiriments', gen_name, f'rouge_global_avgs.json'), idx_avgs_df)


