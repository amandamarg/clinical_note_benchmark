from dotenv import load_dotenv
import pandas as pd
import os
import requester
from tqdm import tqdm
from rouge import Rouge
import json
import re    
from glob import glob
from run import match_filenames

# def add_eval(gen_note, standard_note, evaluators, report={}):
#     rouge = Rouge()
#     report['rouge'] = rouge.get_scores(gen_note, standard_note)
#     for evaluator in evaluators:
#         eval = evaluator.send((gen_note, standard_note))
#         report[evaluator.model_name + '-' + evaluator.prompt_name] = eval
#     return report

# def eval(gen_note_paths_dicts, df, evaluators, standard_dir='standards', full_report={}):
#     df = df.copy()
#     if 'idx' in df.columns:
#         df = df.set_index('idx')
#     for idx,paths in gen_note_paths_dicts.items():
#         if idx not in full_report.keys():
#             full_report[idx] = []
#         for path in paths:
#             with open(path, 'r') as file:
#                 gen_note = file.read()
#             standard_path = os.path.join(standard_dir, f'standard_note_{str(idx)}.txt')
#             if os.path.islink(standard_path):
#                 standard_path = os.readlink(standard_path)
#                 with open(standard_path, 'r') as file:
#                     standard_note = file.read()
#             else:
#                 standard_note = df.loc[int(idx)]['full_note']
#                 standard_path = 'ref'
#             eval_report = {"evaluated_note": path, "standard_note": standard_path, "results": {}}
#             add_eval(gen_note, standard_note, evaluators, eval_report["results"])
#             full_report[idx].append(eval_report)
#     return full_report


def set_standard(idx, src_path, standard_dir='standards'):
    dest_path = os.path.join(standard_dir, f'standard_note_{str(idx)}.txt')
    if (os.path.islink(dest_path)):
        os.unlink(dest_path)
    os.symlink(src_path, dest_path)

def get_gen_note_paths(dir_path, idx=None, version=None, return_matches=False):
    gen_note_paths = glob(os.path.join(dir_path, '**/*.txt'), recursive=True)
    if isinstance(idx, list):
        idx = list(map(str, idx))
        idx_filter = ('|').join(idx)
    else:
        idx_filter = str(idx) if idx else r'\d+'
    if isinstance(version, list):
        version = list(map(str, version))
        if '0' in version:
            version.remove('0')
            version_filter = rf'({('|').join(version)})*'
        else:
            version_filter = ('|').join(version)
    else:
        version_filter = '' if str(version) == '0' else str(version) if version else r'\d*'
    pattern = re.compile(rf'(.*)/({idx_filter})/gen_note({version_filter}).txt')
    if return_matches:
        return [pattern.match(p) for p in gen_note_paths if pattern.match(p)]
    else:
        return [p for p in gen_note_paths if pattern.match(p)]

# def get_standard_path(idx, standard_dir='standards'):
#     standard_path = os.path.join(standard_dir, f'standard_note_{str(idx)}.txt')
#     if os.path.islink(standard_path):
#         return os.readlink(standard_path)
#     else:
#         return 'ref'
def read_gen_notes(gen_note_paths):
    pattern = re.compile(rf'(.*)/(\d+)/gen_note(\d*).txt')
    gen_notes = []
    inds = []
    for p in gen_note_paths:
        match = pattern.match(p)
        assert match
        inds.append(int(match[2]))
        with open(p, 'r') as file:
            gen_notes.append(file.read())
    return pd.DataFrame(list(zip(gen_note_paths, gen_notes, inds)), columns=['path', 'note', 'idx'])
    
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

        
if __name__ == '__main__':
    load_dotenv()
    eval = Evaluator('s1', 'ozwell', os.getenv("OZWELL_SECRET_KEY"))
    samples = sorted(os.listdir('ozwell/g1'), key=int)[:15]
    paths = get_gen_note_paths('ozwell/g1')
    gen_notes_df = read_gen_notes(paths)
    # gen_notes_df['path'] eval.get_standards(gen_notes_df)['path']
    print(gen_notes_df)
    






