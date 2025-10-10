from dotenv import load_dotenv
import pandas as pd
import os
import requester
from evaluate import Evaluator
from generate import generate_loop, generate_n
from utils import get_gen_note_paths, read_gen_notes, set_standard, write_file
import json

if __name__ == '__main__':
    load_dotenv()
    df = pd.read_json('augmented-clinical-notes/augmented_notes_30K.jsonl', lines=True)
    samples = ['224', '431', '562', '619', '958', '1380', '1716', '1834', '2021', '3026', '3058', '3093', '3293', '3931', '4129']
    df_samples = df[df['idx'].astype('string').isin(samples)]
    
    #Ozwell
    req = requester.OzwellRequester("g2", os.getenv("OZWELL_SECRET_KEY"))
    gen_name = req.model_name + '_' + req.prompt_name
    n = 3
    generate_n(n, df_samples, req)
    eval = Evaluator()
    req.set_prompt('s1')
    eval_name = req.model_name + '-' + req.prompt_name
    avgs = {'global':{}, 'indiv':{}}
    for i in range(n):
        gen_note_paths = get_gen_note_paths('ozwell/g2', samples, i)
        gen_notes = read_gen_notes(gen_note_paths)
        standards = eval.get_standards(gen_notes)
        ai_responses = eval.ai_eval(gen_notes, req)
        rouge_scores = eval.get_rouge(gen_notes, False)
        eval_df = pd.DataFrame(list(zip(gen_notes['idx'].values, gen_notes['path'].values, standards['path'].values, ai_responses, rouge_scores)), columns=['idx', 'gen_note_path', 'standard_note_path', eval_name, 'rouge'])
        eval_df.to_json(f'expiriments/standard_ref_{gen_name}_eval_report_{i}.json')
        avgs['global'][f'v{i}'] = eval.get_rouge(gen_notes, True)
    for s in samples:
        gen_note_paths = get_gen_note_paths('ozwell/g2', s)
        gen_notes = read_gen_notes(gen_note_paths)
        avgs['indiv'][s] = eval.get_rouge(gen_notes, True)
    with open(f'expiriments/standard_ref_{gen_name}_rouge_avgs.json', 'w') as file:
        json.dump(avgs, file)
