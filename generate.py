import os
from tqdm import tqdm
from utils import write_file, match_filenames

def generate_loop(df, generator, overwrite=False):
    df = df.copy()
    if 'idx' in df.columns:
        df = df.set_index('idx')
    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=50):
        path = os.path.join(generator.root_dir, generator.model_name, generator.prompt_name, str(idx))
        if not os.path.exists(path):
            os.makedirs(path)
        gen_note = generator.send(row['conversation'])
        write_file(gen_note, path, "gen_note", "txt", overwrite)


def generate_n(n, df, generator):
    df = df.copy()
    if 'idx' in df.columns:
        df = df.set_index('idx')
    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=50):
        path = os.path.join(generator.root_dir, generator.model_name, generator.prompt_name, str(idx))
        if not os.path.exists(path):
            os.makedirs(path)
        versions = match_filenames(path, "gen_note", "txt")
        if len(versions) < n:
            for _ in range(n-len(versions)):
                gen_note = generator.send(row['conversation'])
                write_file(gen_note, path, "gen_note", "txt", False)


