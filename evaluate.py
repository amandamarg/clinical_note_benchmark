import pandas as pd
import os
from tqdm import tqdm
from rouge import Rouge
import requester
import re
from utils import get_standard_path, get_gen_paths, read, get_metadata_df, write_reports


IDXS = [224, 431, 562, 619, 958, 1380, 1716, 1834, 2021, 3026, 3058, 3093, 3293, 3931, 4129] # or 'all'
EVAL_MODEL_NAME = 'ozwell'
EVAL_PROMPT = 's1'
CLEAN_NOTES = False
class Evaluator:
    def __init__(self, requester, prompt_name, clean_notes=False):
        self.clean_notes = clean_notes
        self.rouge = Rouge()
        self.requester = requester
        assert prompt_name[0] == 's'
        self.prompt_name = prompt_name

    def clean_text(self, text):
        text = re.sub(r'(\\n)|(\n)|(-)|(\*\*)', ' ', text)
        return re.sub(r' +', ' ', text).strip()
    
    def ai_eval(self, standard_notes, gen_notes):
        if self.requester.prompt_name != self.prompt_name:
            self.requester.set_prompt(self.prompt_name)
        responses = []
        if self.clean_notes:
            standard_notes = map(self.clean_text, standard_notes)
            gen_notes = map(self.clean_text, gen_notes)
        notes = zip(standard_notes, gen_notes)
        for standard, gen in tqdm(notes, total=len(notes), ncols=50):
            responses.append(self.requester.send((gen, standard)))
        return responses
    

    def get_rouge(self, standard_notes, gen_notes, avg=True):
        if self.clean_notes:
            standard_notes = map(self.clean_text, standard_notes)
            gen_notes = map(self.clean_text, gen_notes)
        return self.rouge.get_scores(gen_notes, standard_notes, avg)
    

if __name__=='__main__':
    for i in IDXS: # use reference note as default standard
        if f'{str(i)}.txt' not in os.listdir('standards'):
            os.symlink(f'results/{str(i)}/full_note.txt', os.path.join('standards', f'{str(i)}.txt'))

    if EVAL_MODEL_NAME == 'ozwell':
        req_eval = requester.OzwellRequester(EVAL_PROMPT)
    else:
        req_eval = requester.OllamaRequester(EVAL_MODEL_NAME, EVAL_PROMPT)
    eval = Evaluator(requester=req_eval, prompt_name=EVAL_PROMPT, clean_notes=CLEAN_NOTES)

    metadata_df = get_metadata_df(get_gen_paths(idxs=IDXS, models='ozwell', prompts='all'))
    standard_paths = metadata_df['idx'].map(get_standard_path)
    gen_note = metadata_df['full_path'].map(read)
    standard_note = map(read, standard_paths)
    model_responses = eval.ai_eval(standard_note, gen_note)
    eval_df = pd.DataFrame(list(zip(metadata_df['idx'], metadata_df['full_path'], gen_note, standard_paths, standard_note, model_responses)), columns=['idx', 'gen_path', 'gen_note', 'standard_path', 'standard_note', EVAL_MODEL_NAME + '-' + EVAL_PROMPT])
    rouge_scores = eval.get_rouge(standard_note, gen_note, avg=False)
    eval_df = pd.concat((eval_df, pd.DataFrame.from_records(rouge_scores)), axis=1)
    for path, df in eval_df.groupby('full_path'):
        write_reports(os.path.dirname(path) + 'eval_report.json', df)

