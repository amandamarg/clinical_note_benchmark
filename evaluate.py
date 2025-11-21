import json
from urllib import response
from dotenv import load_dotenv
from openai import OpenAI
import os
from tqdm import tqdm
from rouge import Rouge
from utils import get_standard_path, search_file_paths, read, get_metadata_df, write_reports, melt_rouge_scores
import re
import seaborn as sns
import pandas as pd


CLEAN_NOTES = False
MODEL = "o4-mini"  # strong tool-calling + reasoning; adjust per your account
OVERWRITE_REPORTS = True  # if True, overwrite existing evaluation reports

with open('tools.json') as f:
    TOOLS = json.load(f)

with open('prompts/evaluation/system_prompt.txt') as f:
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
        
    def eval(self, gen_path, overwrite=False):
        idx = int(gen_path.split('/')[-5])
        standard_path = get_standard_path(idx)
        standard_note = read(standard_path)
        gen_note = read(gen_path)
        metadata = {'standard_note_path': standard_path, 'cleaned': self.clean_notes}
        if self.clean_notes:
            standard_note = self.clean_text(standard_note)
            gen_note = self.clean_text(gen_note)
        rouge_scores = self.rouge.get_scores(gen_note, standard_note, avg=False)
        for key, value in rouge_scores[0].items():
            report_path = os.path.join(os.path.dirname(gen_path), f'{key}.json')
            self.write(report_path, [{**metadata, **value}], overwrite=overwrite)
        metadata.update({'model': self.model})
        model_eval = self.compare_documents(standard_note, gen_note, include_raw=False)
        self.write(os.path.join(os.path.dirname(gen_path), f'ai_eval.json'),  [{**metadata, **model_eval}], overwrite=overwrite)
    
    def write(self, path, data, overwrite=False):
        if os.path.exists(path) and not overwrite:
            with open(path, 'r') as file:
                existing_report = json.load(file)
            existing_report.extend(data)
            with open(path, 'w') as file:
                json.dump(existing_report, file, indent=4)
        else:
            with open(path, 'w') as file:
                json.dump(data, file, indent=4)


if __name__=='__main__':
    # gen_file_paths = search_file_paths(filename='gen_note.txt', idxs='all', models='ozwell', prompts='all')
    gen_file_paths = ['results/155216/ozwell/g2/1763744878.57512/gen_note.txt']
    eval = Evaluator(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        tools=TOOLS,
        clean_notes=CLEAN_NOTES
    )
    for path in tqdm(gen_file_paths):
        eval.eval(path, overwrite=OVERWRITE_REPORTS)