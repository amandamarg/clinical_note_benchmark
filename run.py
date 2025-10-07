from dotenv import load_dotenv
import pandas as pd
import os
import requester
from tqdm import tqdm
from rouge import Rouge
import json
import re    

def match_filenames(dir_path, base_filename, extension):
    file_pattern = re.compile(rf'{base_filename}(\d*).{extension}')
    matches = [re.match(file_pattern, f) for f in os.listdir(dir_path) if re.match(file_pattern, f)]
    return matches
        
def write_file(contents, dir_path, base_filename, extension, overwrite=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    versions = [0 if (m[1] == '') else int(m[1]) for m in match_filenames(dir_path, base_filename, extension)]
    if overwrite or len(versions) == 0:
        file_path = os.path.join(dir_path, f'{base_filename}.{extension}')
    else:
        file_path = os.path.join(dir_path, f'{base_filename}{max(versions) + 1}.{extension}')
    with open(file_path, 'w') as file:
        if extension == "json":
            json.dump(contents, file)
        else:
            file.write(contents)
    return file_path
        
def add_eval(gen_note, standard_note, evaluators, report={}):
    rouge = Rouge()
    report['rouge'] = rouge.get_scores(gen_note, standard_note)
    for evaluator in evaluators:
        eval = evaluator.send((gen_note, standard_note))
        if evaluator.model_name not in report.keys():
            report[evaluator.model_name] = {}
        report[evaluator.model_name][evaluator.prompt_name] = eval
    return report

def build_eval_report(gen_notes, standard_notes, evaluators):
    report = {}
    for g_name, g_note in gen_notes.items():
        report[g_name] = {}
        for s_name, s_note in standard_notes.items():
            if s_name != g_name:
                report[g_name][s_name] = {}
                add_eval(g_note, s_note, evaluators, report[g_name][s_name])
    return report

def eval_dir(dir_path, ref_note, evaluators, overwrite=False, standards='ref'):
    standard_notes = {}
    gen_notes = {}
    for f in os.scandir(dir_path):
        if re.match(r'gen_note(\d+).txt', f.name):
            with open(f.path, 'r') as file:
                gen_notes[f.name] = file.read()
    if isinstance(standards, list):
        for s in standards:
            if s in gen_notes.keys():
                standard_notes[s] = gen_notes[s]
    elif standards == 'ref':
        standard_notes = {'ref': ref_note}
    elif standards == 'all':
        standard_notes = {k:v for k,v in gen_notes}
        standard_notes['ref'] = ref_note
    else:
        raise ValueError
    assert len(standard_notes) > 0
    report = build_eval_report(gen_notes, standard_notes, evaluators)
    write_file(report, dir_path, 'eval_report', 'json', overwrite)

def generate_loop(df, generator, overwrite=False):
    df = df.copy()
    if 'idx' in df.columns:
        df = df.set_index('idx')
    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=50):
        path = os.path.join(generator.root_dir, generator.model_name, generator.prompt_name, str(idx))
        gen_note = generator.send(row['conversation'])
        write_file(gen_note, path, "gen_note", "txt", overwrite)

def eval_loop(dir_path, df, evaluators, overwrite=False, standards='ref'):
    df = df.copy()
    if 'idx' in df.columns:
        df = df.set_index('idx')
    for d in tqdm(os.scandir(dir_path), total=len(os.listdir(dir_path)), ncols=50):
        eval_dir(d.path, df.loc[int(d.name)]['full_note'], evaluators, overwrite, standards)

def gen_eval(df, generator, evaluators, overwrite=False, standards='ref'):
    df = df.copy()
    if 'idx' in df.columns:
        df = df.set_index('idx')
    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=50):
        path = os.path.join(generator.root_dir, generator.model_name, generator.prompt_name, str(idx))
        gen_note = generator.send(row['conversation'])
        write_file(gen_note, path, "gen_note", "txt", overwrite)
        eval_dir(path, row['full_note'], evaluators, overwrite, standards)

if __name__ == '__main__':
    load_dotenv()
    print("Loading data...")
    df = pd.read_json("hf://datasets/AGBonnet/augmented-clinical-notes/augmented_notes_30K.jsonl", lines=True)
    print("Done loading data!")

    #Ozwell
    # generator = requester.OzwellRequester("g1", os.getenv("OZWELL_SECRET_KEY"))
    # evaluator = requester.OzwellRequester("s1", os.getenv("OZWELL_SECRET_KEY"))
    # main(df, generator, [evaluator])
    
    #Gemma3
    # generator = requester.OllamaRequester("gemma3", "g1")
    # evaluator = requester.OzwellRequester("gemma3", "s1")
    # main(df, generator, [evaluator])


