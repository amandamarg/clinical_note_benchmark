from utils import parse_path
from glob import glob
import pandas as pd


if __name__ == '__main__':
    df = []

    for path in glob('**/ai_eval.json', recursive=True):
        path_data = parse_path(path)
        df.append(pd.read_json(path).assign(**path_data))

    df = pd.concat(df, ignore_index=True)
    missing = pd.concat((df['full_path'].str.replace('ai_eval.json', 'gen_note.txt'), df['idx'], df['prompt'], pd.json_normalize(df['missing'].explode())), axis=1)
    added = pd.concat((df['full_path'].str.replace('ai_eval.json', 'gen_note.txt'), df['idx'], df['prompt'], pd.json_normalize(df['added'].explode())), axis=1)

    #look for differences between missing clinical concepts across runs for each prompt, idx combination
    missing_clinical_concepts = []
    for (idx, prompt), group in missing.groupby(['idx', 'prompt']):
        if len(group['clinical_concept'].explode().unique()) > 1:
            missing_clinical_concepts.append((idx, prompt, group['clinical_concept'].explode().unique().tolist()))
    missing_clinical_concepts = pd.DataFrame(missing_clinical_concepts, columns=['idx', 'prompt', 'missing_clinical_concepts'])
    missing_clinical_concepts.to_json('missing_clinical_concepts.json', orient='records', indent=4)

    #look for differences between added clinical concepts across runs for each prompt, idx combination
    added_clinical_concepts = []
    for (idx, prompt), group in added.groupby(['idx', 'prompt']):
        if len(group['clinical_concept'].explode().unique()) > 1:
            added_clinical_concepts.append((idx, prompt, group['clinical_concept'].explode().unique().tolist()))

    added_clinical_concepts = pd.DataFrame(added_clinical_concepts, columns=['idx', 'prompt', 'added_clinical_concepts'])
    added_clinical_concepts.to_json('added_clinical_concepts.json', orient='records', indent=4)
