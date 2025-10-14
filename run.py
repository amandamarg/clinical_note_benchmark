from dotenv import load_dotenv
import pandas as pd
import os
import requester
from evaluate import Evaluator, get_eval_df, get_rouge_avgs_df
from generate import generate_loop, generate_n
from utils import get_gen_note_paths, read_gen_notes, set_standard, write_df
import json
import re
from utils import match_filenames, extend_file, write_file, get_most_recent
from glob import glob

N = 3
WRITE_MODE = 'extend' # 'extend', 'overwrite', 'new'
STANDARD_VERSION = '2'
EXCLUDE_STANDARDS_FROM_AVG = True
GENERATE_MODEL_NAME = 'ozwell'
GENERATE_PROMPT_NAME = 'g1'
EVAL_MODEL_NAME = 'ozwell'
EVAL_PROMPT = 's1'
CLEAN_NOTES = False

def main(samples = ['224', '431', '562', '619', '958', '1380', '1716', '1834', '2021', '3026', '3058', '3093', '3293', '3931', '4129']):
    df = pd.read_json('augmented-clinical-notes/augmented_notes_30K.jsonl', lines=True)
    df_samples = df[df['idx'].astype('string').isin(samples)]
    
    if GENERATE_MODEL_NAME == 'ozwell':
        req_gen = requester.OzwellRequester(GENERATE_PROMPT_NAME)
    else:
        req_gen = requester.OllamaRequester(GENERATE_MODEL_NAME, GENERATE_PROMPT_NAME)
    
    if EVAL_MODEL_NAME == 'ozwell':
        req_eval = requester.OzwellRequester(EVAL_PROMPT)
    else:
        req_eval = requester.OllamaRequester(EVAL_MODEL_NAME, EVAL_PROMPT)

    base_path = os.path.join('expiriments', req_gen.model_name, req_gen.prompt_name)
    for s in samples:
        if STANDARD_VERSION != 'ref':
            set_standard(s, os.path.join(GENERATE_MODEL_NAME, GENERATE_PROMPT_NAME, str(s), f'gen_note{str(STANDARD_VERSION) if str(STANDARD_VERSION) != '0' else ''}.txt'))
        else:
            set_standard(s, 'ref')
    generate_n(N, df_samples, req_gen)
    eval = Evaluator()
    gen_note_paths = glob(f'{req_gen.model_name}/{req_gen.prompt_name}/*/gen_note*.txt', recursive=True)
    gen_note_paths = [path for path in gen_note_paths if path.split('/')[-2] in samples]
    gen_notes = read_gen_notes(gen_note_paths)

    eval_df = get_eval_df(gen_notes, eval, req_eval)
    if EXCLUDE_STANDARDS_FROM_AVG:
        gen_notes = gen_notes[~gen_notes['path'].isin(eval.standards['path'])]
    rouge_avgs_df = get_rouge_avgs_df(gen_notes, eval)

    write_df(eval_df, base_path, 'eval_report', 'json', WRITE_MODE)
    write_df(rouge_avgs_df, base_path, 'rouge_avgs', 'json', WRITE_MODE)

if __name__ == '__main__':
    main()

