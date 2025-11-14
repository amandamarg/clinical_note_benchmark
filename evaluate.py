import json
from dotenv import load_dotenv
from openai import OpenAI
import os
from tqdm import tqdm
from rouge import Rouge
from utils import get_standard_path, search_file_paths, read, get_metadata_df, write_reports, melt_rouge_scores
import re
import seaborn as sns
import pandas as pd

# IDXS = [224, 431, 562, 619, 958, 1380, 1716, 1834, 2021, 3026, 3058, 3093, 3293, 3931, 4129] # or 'all'
IDXS = [155216]
INCLUDE_INDIV_PLOTS = False # if True, generate and save ROUGE score plots for each evaluation
CLEAN_NOTES = False
MODEL = "o4-mini"  # strong tool-calling + reasoning; adjust per your account
OVERWRITE_REPORTS = True  # if True, overwrite existing evaluation reports
REPORT_NAME = 'eval_report'

with open('tools.json') as f:
    TOOLS = json.load(f)

with open('prompts/system_prompt.txt') as f:
    SYSTEM_PROMPT = f.read()

class Evaluator:
    def __init__(self, model, system_prompt, tools, clean_notes=False):
        load_dotenv()
        self.model = model
        self.clean_notes = clean_notes
        self.rouge = Rouge()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = system_prompt
        self.tools = tools
        
    def clean_text(self, text):
        text = re.sub(r'(\\n)|(\n)|(-)|(\*\*)', ' ', text)
        return re.sub(r' +', ' ', text).strip()

    def compare_documents(self, doc_a: str, doc_b: str, include_raw=False):
        input_list = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Compare Document B to Document A and act per the rules."},
                    {"type": "input_text", "text": doc_a},
                    {"type": "input_text", "text": doc_b},
                ]
            }
        ]

        # Make the call. Tool calls (if any) will appear in response.output with type='tool_call'.
        resp = self.client.responses.create(
            model=self.model,
            input=input_list,
            tools=self.tools,
            tool_choice="auto"
        )

        # Handle tool calls
        added, missing = [], []
        # The Responses API can return a list of output items; iterate and execute tools accordingly.
        for item in resp.output:
            if item.type == "function_call":
                args = json.loads(item.arguments)
                if item.name == "report_added_doc":
                    added.append(args)
                elif item.name == "report_missing_doc":
                    missing.append(args)

        # If there were no tool calls, capture any text the model returned
        text_output = "\n".join([i.content[0].text for i in resp.output if getattr(i, "type", None) == "message"])
        if include_raw:
            return {
                "added": added,
                "missing": missing,
                "text": text_output.strip() or None,
                "raw": resp  # keep for audit if needed
            }
        else:
            return {
                "added": added,
                "missing": missing,
                "text": text_output.strip() or None
            }
        
    
    def ai_eval(self, standard_notes, gen_notes):
        responses = []
        if self.clean_notes:
            standard_notes = list(map(self.clean_text, standard_notes))
            gen_notes = list(map(self.clean_text, gen_notes))
        notes = list(zip(standard_notes, gen_notes))
        for standard, gen in tqdm(notes, total=len(notes), ncols=50):
            responses.append(self.compare_documents(standard, gen))
        return responses

    def get_rouge(self, standard_notes, gen_notes, avg=True):
        if self.clean_notes:
            standard_notes = list(map(self.clean_text, standard_notes))
            gen_notes = list(map(self.clean_text, gen_notes))
        return self.rouge.get_scores(gen_notes, standard_notes, avg)



if __name__=='__main__':
    metadata_df = get_metadata_df(search_file_paths(filename='gen_note.txt', idxs=IDXS, models='ozwell', prompts='all'))
    standard_paths = metadata_df['idx'].map(get_standard_path)
    gen_note = list(metadata_df['full_path'].map(read))
    standard_note = list(map(read, standard_paths))
    eval = Evaluator(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        tools=TOOLS,
        clean_notes=CLEAN_NOTES
    )
    model_responses = eval.ai_eval(standard_note, gen_note)
    eval_df = pd.DataFrame(list(zip(metadata_df['idx'], metadata_df['full_path'], gen_note, standard_paths, standard_note, model_responses)), columns=['idx', 'gen_note_path', 'gen_note', 'standard_note_path', 'standard_note', MODEL + '_response'])
    rouge_scores = eval.get_rouge(standard_note, gen_note, avg=False)
    eval_df = pd.concat((eval_df, pd.DataFrame.from_records(rouge_scores)), axis=1)
    for path, df in eval_df.groupby('gen_note_path'):
        write_reports(os.path.dirname(path) + f'/{REPORT_NAME}.json', df, overwrite=OVERWRITE_REPORTS)
        if INCLUDE_INDIV_PLOTS:
            melted = melt_rouge_scores(df)
            g = sns.catplot(melted, col='metric', x='rouge_type', y='value', hue='standard_note_path', dodge=True, legend='full')
            g.savefig(os.path.join(os.path.dirname(path), 'rouge_plot.png'))
    
