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

def read_files(dir_path, base_filename, extension):
    all_files = {}
    for f in os.scandir(dir_path):
        if re.match(rf'{base_filename}(\d*{extension}', f.name):
            with open(f.path, 'r') as file:
                if extension == 'json':
                    all_files[f.name] = json.load(file)
                else:
                    all_files[f.name] = file.read()
    return all_files
        
def add_eval(gen_note, standard_note, evaluators, report={}):
    rouge = Rouge()
    report['rouge'] = rouge.get_scores(gen_note, standard_note)
    for evaluator in evaluators:
        eval = evaluator.send((gen_note, standard_note))
        report[evaluator.model_name + '-' + evaluator.prompt_name] = eval
    return report

def generate_loop(df, generator, overwrite=False):
    df = df.copy()
    if 'idx' in df.columns:
        df = df.set_index('idx')
    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=50):
        path = os.path.join(generator.root_dir, generator.model_name, generator.prompt_name, str(idx))
        gen_note = generator.send(row['conversation'])
        write_file(gen_note, path, "gen_note", "txt", overwrite)

def eval(gen_note_paths_dicts, df, evaluators, standard_dir='standards', full_report={}):
    df = df.copy()
    if 'idx' in df.columns:
        df = df.set_index('idx')
    for idx,paths in gen_note_paths_dicts.items():
        if idx not in full_report.keys():
            full_report[idx] = []
        for path in paths:
            with open(path, 'r') as file:
                gen_note = file.read()
            standard_path = os.path.join(standard_dir, f'standard_note_{str(idx)}.txt')
            if os.path.islink(standard_path):
                standard_path = os.readlink(standard_path)
                with open(standard_path, 'r') as file:
                    standard_note = file.read()
            else:
                standard_note = df.loc[int(idx)]['full_note']
                standard_path = 'ref'
            eval_report = {"evaluated_note": path, "standard_note": standard_path, "results": {}}
            add_eval(gen_note, standard_note, evaluators, eval_report["results"])
            full_report[idx].append(eval_report)
    return full_report

def gen_eval(df, generator, evaluators, overwrite=False, standard_dir='standards'):
    df = df.copy()
    if 'idx' in df.columns:
        df = df.set_index('idx')
    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=50):
        path = os.path.join(generator.root_dir, generator.model_name, generator.prompt_name, str(idx))
        gen_note = generator.send(row['conversation'])
        gen_note_path = write_file(gen_note, path, "gen_note", "txt", overwrite)
        standard_path = os.path.join(standard_dir, f'standard_note_{str(idx)}.txt')
        if os.path.islink(standard_path):
            standard_path = os.readlink(standard_path)
            with open(standard_path, 'r') as file:
                standard_note = file.read()
        else:
            standard_note = row['full_note']
            standard_path = 'ref'
        eval_report = {"evaluated_note": gen_note_path, "standard_note": standard_path, "results": {}}
        add_eval(gen_note, standard_note, evaluators, eval_report["results"])
        write_file(eval_report, path, 'eval_report', 'json', overwrite)

def set_standard(idx, src_path, standard_dir='standards'):
    dest_path = os.path.join(standard_dir, f'standard_note_{str(idx)}.txt')
    if (os.path.islink(dest_path)):
        os.unlink(dest_path)
    os.symlink(src_path, dest_path)

def generate_n(n, df, generator):
    df = df.copy()
    if 'idx' in df.columns:
        df = df.set_index('idx')
    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=50):
        path = os.path.join(generator.root_dir, generator.model_name, generator.prompt_name, str(idx))
        versions = match_filenames(path, "gen_note", "txt")
        if len(versions) < n:
            for _ in range(n-len(versions)):
                gen_note = generator.send(row['conversation'])
                write_file(gen_note, path, "gen_note", "txt", False)

def avg_rouge(rouge_scores):
    avgs = {}
    for x in rouge_scores:
        for k,v in x.items():
            if k not in avgs.keys():
                avgs[k] = {}
            for kk,vv in v.items():
                if kk not in avgs[k].keys():
                    avgs[k][kk] = []
                avgs[k][kk].append(vv)
    for k,v in avgs.items():
        for kk,vv in v.items():
            avgs[k][kk] = sum(vv)/len(vv)
    return avgs



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

    #Expiriment
    samples = sorted(os.listdir('ozwell/g1'), key=int)[:15]
    df_subset = df[df['idx'].astype('string').isin(samples)]
    req = requester.OzwellRequester("g1", os.getenv("OZWELL_SECRET_KEY"))
    # n = 3
    # generate_n(n, df_subset, req)
    base_path = os.path.join(req.root_dir, req.model_name, req.prompt_name)
    # req.set_prompt('s1')
    # gen_note_paths = {}
    # for idx in samples:
    #     gen_note_paths[idx] = []
    #     for f in os.scandir(os.path.join(base_path, str(idx))):
    #         if re.match(r'gen_note(\d*).txt', f.name):
    #             gen_note_paths[idx].append(f.path)
    # report = {}
    # eval(gen_note_paths, df_subset, [req], '', report)
    # for i in range(n):
    #     for k,v in gen_note_paths.items():
    #         set_standard(k,v[i],'standards')
    #     eval(gen_note_paths, df_subset, [req], 'standards', report)
    # write_file(report, 'expiriments', 'eval_report', 'json', False)

    notes = {'ref':[]}
    for idx in samples:
        notes['ref'].append(df_subset[df_subset['idx'] == int(idx)]['full_note'].values[0])
        for f in os.scandir(os.path.join(base_path, str(idx))):
            if re.match(r'gen_note(\d*).txt', f.name):
                if f.name not in notes.keys():
                    notes[f.name] = []
                with open(f.path, 'r') as file:
                    notes[f.name].append(file.read())
    
    rouge = Rouge()
    avgs = []
    for s_k, s_v in notes.items():
        for g_k, g_v in notes.items():
            if g_k != 'ref' and g_k != s_k:
                scores = rouge.get_scores(g_v, s_v, avg=True)
                avgs.append({'standard': s_k, "gen_note": g_k, "results": scores})
    write_file(avgs, 'expiriments', 'rouge_avgs', 'json', False)
        

        
    

    

