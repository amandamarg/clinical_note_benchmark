import pandas as pd
import os
from tqdm import tqdm
from rouge import Rouge
import requester
import re
from utils import get_standard_path, search_file_paths, read, get_metadata_df, write_reports, melt_rouge_scores
import seaborn as sns
'''
Run this script to evaluate generated clinical notes against standard notes based on ROUGE scores and AI model evaluations with a specified model and similarity prompt.
Optionally, standard notes can be cleaned before evaluation.
***Make sure to set standard notes using set_standards.py before running this script.***
'''
IDXS = [224, 431, 562, 619, 958, 1380, 1716, 1834, 2021, 3026, 3058, 3093, 3293, 3931, 4129] # or 'all'
EVAL_MODEL_NAME = 'ozwell'
EVAL_PROMPT = 's1'
CLEAN_NOTES = False
INCLUDE_INDIV_PLOTS = False # if True, generate and save ROUGE score plots for each evaluation
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
    if EVAL_MODEL_NAME == 'ozwell':
        req_eval = requester.OzwellRequester(EVAL_PROMPT)
    else:
        req_eval = requester.OllamaRequester(EVAL_MODEL_NAME, EVAL_PROMPT)
    eval = Evaluator(requester=req_eval, prompt_name=EVAL_PROMPT, clean_notes=CLEAN_NOTES)

    metadata_df = get_metadata_df(search_file_paths(filename='gen_note.txt', idxs=IDXS, models='ozwell', prompts='all'))
    standard_paths = metadata_df['idx'].map(get_standard_path)
    gen_note = metadata_df['full_path'].map(read)
    standard_note = map(read, standard_paths)
    model_responses = eval.ai_eval(standard_note, gen_note)
    eval_df = pd.DataFrame(list(zip(metadata_df['idx'], metadata_df['full_path'], gen_note, standard_paths, standard_note, model_responses)), columns=['idx', 'gen_note_path', 'gen_note', 'standard_note_path', 'standard_note', EVAL_MODEL_NAME + '-' + EVAL_PROMPT])
    rouge_scores = eval.get_rouge(standard_note, gen_note, avg=False)
    eval_df = pd.concat((eval_df, pd.DataFrame.from_records(rouge_scores)), axis=1)
    for path, df in eval_df.groupby('full_path'):
        write_reports(os.path.dirname(path) + 'eval_report.json', df)
        if INCLUDE_INDIV_PLOTS:
            melted = melt_rouge_scores(df)
            g = sns.catplot(melted, col='metric', x='rouge_type', y='value', hue='standard_note_path', dodge=True, legend='full')
            g.savefig(os.path.join(os.path.dirname(path), 'rouge_plot.png'))


