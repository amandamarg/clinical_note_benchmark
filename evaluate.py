import pandas as pd
import os
from tqdm import tqdm
from rouge import Rouge
from utils import match_filenames, read_gen_notes, set_standard
import requester
from glob import glob
import re
class Evaluator:
    def __init__(self, dataset_path='augmented-clinical-notes/augmented_notes_30K.jsonl', standards_dir='standards', clean_notes=False):
        self.clean_notes = clean_notes
        self.init_standards(dataset_path, standards_dir)
        self.rouge = Rouge()

    def init_standards(self, dataset_path, standard_dir):
        matches = match_filenames(standard_dir, 'standard_note_', 'txt')
        standard_inds = [int(m[1]) for m in matches]
        standard_paths = [os.readlink(os.path.join(standard_dir, m[0])) for m in matches]
        standard_notes = []
        for p in standard_paths:
            with open(p, 'r') as file:
                standard_notes.append(file.read())
        standard_df = pd.DataFrame(list(zip(standard_paths, standard_notes)), columns=['path', 'full_note'], index=standard_inds)
        df = pd.read_json(dataset_path, lines=True)
        df = df.set_index('idx')
        df = df[~df.index.isin(standard_inds)].get(['full_note']).copy()
        df['path'] = df.map(lambda x: 'ref')
        if len(standard_df) == 0:
            self.standards = df
        elif len(df) == 0:
            self.standards = standard_df
        else:
            self.standards = pd.concat((df, standard_df))
        if self.clean_notes:
            self.standards['full_note'] = self.standards['full_note'].map(self.clean_text)
            

    def clean_text(self, text):
        text = re.sub(r'(\\n)|(\n)|(-)|(\*\*)', ' ', text)
        return re.sub(r' +', ' ', text).strip()
    
    def ai_eval(self, gen_note_df, ai_requester):
        assert ai_requester.prompt_name[0] == 's'
        responses = []
        if self.clean_notes:
            gen_note_df = gen_note_df.copy()
            gen_note_df['note'] = gen_note_df['note'].map(self.clean_text)
        for row in tqdm(gen_note_df.iterrows(), total=len(gen_note_df), ncols=50):
            standard_note = self.standards.loc[row[1]['idx']]['full_note']
            gen_note = row[1]['note']

            responses.append(ai_requester.send((gen_note, standard_note)))
        return responses

    def get_rouge(self, gen_note_df, avg=True):
        standards = self.standards.loc[gen_note_df['idx'].astype(int)]
        if self.clean_notes:
            gen_note_df = gen_note_df.copy()
            gen_note_df['note'] = gen_note_df['note'].map(self.clean_text)
        return self.rouge.get_scores(gen_note_df['note'], standards['full_note'], avg)
    

def get_rouge_avgs_df(gen_notes, eval):
    rouge_avgs = []
    standards = eval.standards
    for i,g in gen_notes.groupby('idx'):
        data = {'standard_note_path': standards.loc[i]['path'], 'idx': i}
        data.update(eval.get_rouge(g, True))
        rouge_avgs.append(data)
    rouge_avgs_df = pd.DataFrame.from_records(rouge_avgs)
    return rouge_avgs_df


def get_eval_df(gen_notes, eval, req):
    eval_data = []
    standards = eval.standards.loc[gen_notes['idx'].astype(int)]
    ai_responses = eval.ai_eval(gen_notes, req)
    eval_data.extend(list(zip(gen_notes['idx'].values, gen_notes['path'].values, standards['path'].values, ai_responses)))
    eval_df = pd.DataFrame(eval_data, columns=['idx', 'gen_note_path', 'standard_note_path', req.model_name + '-' + req.prompt_name])
    rouge_scores = eval.get_rouge(gen_notes, False)
    eval_df = pd.concat((eval_df, pd.DataFrame.from_records(rouge_scores)), axis=1)
    return eval_df

if __name__=='__main__':
    df = pd.read_json('augmented-clinical-notes/augmented_notes_30K.jsonl', lines=True)
    samples = ['224', '431', '562', '619', '958', '1380', '1716', '1834', '2021', '3026', '3058', '3093', '3293', '3931', '4129']
    # samples = ['224', '431', '562', '619', '958', '1380', '1716', '1834', '2021', '3026', '3058', '3093', '3293', '3931', '4129']
    for s in samples:
        set_standard(s, 'ref')
    df_samples = df[df['idx'].astype('string').isin(samples)]
    eval = Evaluator(clean_notes=True)
    req = requester.OzwellRequester('s1')
    gen_note_paths = glob(f'ozwell/g2/*/gen_note*.txt', recursive=True)
    gen_note_paths = [path for path in gen_note_paths if path.split('/')[-2] in samples]
    gen_notes = read_gen_notes(gen_note_paths)
    print(get_eval_df(gen_notes, eval, req))
    
    # x = pd.read_json('expiriments/ozwell/g2/eval_report.json').drop_duplicates(['gen_note_path', 'standard_note_path'])
    # x = x[x['standard_note_path'].isin(eval.standards['path'])]
    # x.set_index('gen_note_path', inplace=True)
    # ai_responses = x.loc[gen_notes['path']]['ozwell-s1']
    # eval_df = get_eval_df(gen_notes, eval, req, ai_responses)


    
