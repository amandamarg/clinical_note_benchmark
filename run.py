from dotenv import load_dotenv
import pandas as pd
import os
import requester
from tqdm import tqdm
from rouge import Rouge
import json

def generate(idx, transcript, generator, overwrite=False):
    assert generator.prompt_name[0] == 'g'
    path = os.path.join(generator.root_dir, generator.model_name, generator.prompt_name, str(idx))
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, "gen_note.txt")
    if os.path.exists(path) and not overwrite:
        with open(path, 'r') as file:
            gen_note = file.read()
        return gen_note, path
    gen_note = generator.send(transcript)
    with open(path, 'w') as file:
        file.write(gen_note)
    return gen_note, path

def generate_all(df, generator, overwrite=False):
    df = df.copy()
    if 'idx' in df.columns:
        df = df.set_index('idx')
    path = os.path.join(generator.root_dir, generator.model_name, generator.prompt_name)
    if not overwrite:
        subdirs = os.listdir(path)
        df = df[~df.index.astype('string').isin(subdirs)]
    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=50):
        generate(idx, row['conversation'], overwrite)
        

def eval(gen_note, ref_note, evaluator, report):
    assert evaluator.prompt_name[0] == 's'
    eval = evaluator.send((gen_note, ref_note))
    if evaluator.model_name not in report.keys():
        report[evaluator.model_name] = {}
    report[evaluator.model_name][evaluator.prompt_name] = eval

def write_eval_reports(dir_path, df, evaluators, overwrite=False):
    df = df.copy()
    if 'idx' in df.columns:
        df = df.set_index('idx')
    n = os.listdir(dir_path)
    rouge = Rouge()
    for f in tqdm(os.scandir(dir_path), total=n, ncols=50):
        idx = int(f.name)
        path = os.path.join(f.path, "eval_report.json")
        if overwrite or not os.path.exists(path):
            report = {}
            with open(os.path.join(f.path, "gen_note.txt"), 'r') as file:
                gen_note = file.read()
            ref_note = df.loc[idx]['full_note']
            report['rouge'] = rouge.get_scores(gen_note, ref_note)
            for e in evaluators:
                eval(gen_note, ref_note, e, report)
            with open(path, 'w') as file:
                json.dump(report, file)
        
def generate_and_eval(df, generator, evaluators, overwrite=False):
    df = df.copy()
    if 'idx' in df.columns:
        df = df.set_index('idx')
    rouge = Rouge()
    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=50):
        gen_note, path = generate(idx, row['conversation'], generator, overwrite)     
        path = os.path.join(os.path.dirname(path), "eval_report.json")
        if not os.path.exists(path) or overwrite:
            ref_note = row['full_note']
            report = {}
            report['rouge'] = rouge.get_scores(gen_note, ref_note)
            for e in evaluators:
                eval(gen_note, ref_note, e, report)
            with open(path, 'w') as file:
                json.dump(report, file)

if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv("OZWELL_SECRET_KEY")
    print("Loading data frame...")
    df = pd.read_json("hf://datasets/AGBonnet/augmented-clinical-notes/augmented_notes_30K.jsonl", lines=True)
    print("Done loading data frame")
    #Ozwell
    # req = requester.OzwellRequester("g1", api_key)
    # path = os.path.join(req.root_dir, req.model_name, req.prompt_name)
    # df.set_index('idx')
    # generate_all(df, req, False)
    # path = os.path.join(req.model_name, req.prompt_name)
    # req.set_prompt('s1')
    # write_eval_reports(path, df, [req], False)

    #Gemma3
    # req = requester.OllamaRequester("gemma3", "g1")
    # path = os.path.join(req.root_dir, req.model_name, req.prompt_name)
    # df.set_index('idx')
    # generate_all(df, req, False)
    # path = os.path.join(req.model_name, req.prompt_name)
    # req.set_prompt('s1')
    # write_eval_reports(path, df, [req], False)

    df_subset = df[df['idx'].astype('string').isin(os.listdir('ozwell/g1'))]
    ol_gen = requester.OllamaRequester("gemma3", "g1")
    ol_sim = requester.OllamaRequester("gemma3", "s1")
    generate_and_eval(df_subset, ol_gen, [ol_sim])

    # add Gemma3 s1 eval to existing Ozwell eval reports
    # req = requester.OllamaRequester("gemma3", "s1")
    # for f in os.scandir("ozwell/g1"):
    #     idx = int(f.name)
    #     eval_report_path = os.path.join(f.path, "eval_report.json")
    #     if os.path.exists(eval_report_path):
    #         with open(eval_report_path, 'r') as file:
    #             eval_report = json.load(file)
    #     if not req.model_name in eval_report.keys() or not "s1" in eval_report[req.model_name].keys():
    #         with open(os.path.join(f.path, "gen_note.txt"), 'r') as file:
    #             gen_note = file.read()
    #         ref_note = df[df['idx'] == idx]['full_note']
    #         eval_report['gemma3']['s1'] == eval(gen_note, ref_note, req, eval_report)
    #         with open(eval_report_path, '1') as file:
    #             json.dump(eval_report, file)