from dotenv import load_dotenv
import pandas as pd
import os
import requester
from tqdm import tqdm
from rouge import Rouge
import json
import re    
from glob import glob


def match_filenames(dir_path, base_filename, extension):
    file_pattern = re.compile(rf'{base_filename}(\d*).{extension}')
    matches = [re.match(file_pattern, f) for f in os.listdir(dir_path) if re.match(file_pattern, f)]
    return matches

def get_most_recent(file_dir, base_filename, extension):
    versions = [0 if (m[1] == '') else int(m[1]) for m in match_filenames(file_dir, base_filename, extension)]
    if len(versions) == 0:
        return None
    version = max(versions)
    file_path = os.path.join(file_dir, f'{base_filename}{version if version != 0 else ""}.{extension}')
    return file_path
        
def write_file(contents, dir_path, base_filename, extension, overwrite=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    versions = [0 if (m[1] == '') else int(m[1]) for m in match_filenames(dir_path, base_filename, extension)]
    if len(versions) == 0:
        file_path = os.path.join(dir_path, f'{base_filename}.{extension}')
    elif overwrite:
        file_path = os.path.join(dir_path, f'{base_filename}{max(versions)}.{extension}')
    else:
        file_path = os.path.join(dir_path, f'{base_filename}{max(versions) + 1}.{extension}')
    if isinstance(contents, pd.DataFrame):
        contents.to_json(file_path)
        return file_path
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
        
def set_standard(idx, src_path, standard_dir='standards'):
    dest_path = os.path.join(standard_dir, f'standard_note_{str(idx)}.txt')
    if (os.path.islink(dest_path)):
        os.unlink(dest_path)
    if src_path == 'ref':
        return
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
            if len(version) == 0:
                version_filter = ''
            else:
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


def extend_file(dest_path, df):
    if os.path.exists(dest_path):
        existing_report = pd.read_json(dest_path)
        pd.concat((existing_report, df)).reset_index(drop=True).to_json(dest_path)
    else:
        df.to_json(dest_path)

def write_df(df, base_path, base_filename, extension, mode):
    path = get_most_recent(base_path, base_filename, extension)
    if path is None:
        path = os.path.join(base_path, f"{base_filename}.{extension}")
        df.to_json(path)
        return path
    if mode == 'overwrite':
        df.to_json(path)
    elif mode == 'extend':
        existing_report = pd.read_json(path)
        pd.concat((existing_report, df)).reset_index(drop=True).to_json(path)
    else:
        v = re.search(rf'{base_filename}(\d*).{extension}', path)
        v = 1 if v[1] == '' else int(v[1]) + 1
        path = os.path.join(base_path, f"{base_filename}{v}.{extension}")
        df.to_json(path)
    return path
    

def clean_text(text):
    text = re.sub(r'(\n)|(-)|(\*\*)', '', text)
    return re.sub(r' +', ' ', text)

