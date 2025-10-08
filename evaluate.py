import pandas as pd
import os
from tqdm import tqdm
from rouge import Rouge
from utils import match_filenames

class Evaluator:
    def __init__(self, dataset_path='augmented-clinical-notes/augmented_notes_30K.jsonl', standards_dir='standards'):
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
        self.standards = pd.concat((df, standard_df))
    
    def get_standards(self, gen_note_df):
        return self.standards.loc[gen_note_df['idx']]

    def ai_eval(self, gen_note_df, ai_requester):
        assert ai_requester.prompt_name[0] == 's'
        responses = []
        for row in tqdm(gen_note_df.iterrows(), total=len(gen_note_df), ncols=50):
            standard = self.standards.loc[row['idx']]
            responses.append(ai_requester.send((row['note'], standard)))
        return responses

    def get_rouge(self, gen_note_df, avg=True):
        standards = self.standards.loc[gen_note_df['idx']]
        return self.rouge.get_scores(gen_note_df['note'], standards['full_note'], avg)





