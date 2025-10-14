# Clinical Note Benchmark

This repository contains tooling for generating clinical notes from patient-doctor conversations, evaluating generated notes against reference (ground-truth) notes using ROUGE and model-based similarity prompts, and collecting results and plots for analysis. This project aims to create a benchmark for assessing the quality of generated clinical notes. This benchmark will serve as a baseline to compare clinical note quality across different models and model versions.

## Contents

- `prompts/` - prompt templates used for generation (`prompts/generation/*.txt`) and similarity/evaluation (`prompts/similarity/*.txt`). Files are named with a leading `g` (generation) or `s` (similarity) plus an identifier, e.g. `g1.txt`, `s1.txt`.
- `ozwell/`, `gemma3/`, etc. - folders where generated notes are stored following the layout `{model}/{prompt}/{idx}/gen_note.txt` and `eval_report.json`.
- `augmented-clinical-notes/` - dataset files (e.g. `augmented_notes_30K.jsonl`) used to provide reference notes and transcripts.
- `expiriments/` - experiment outputs, aggregated ROUGE averages and evaluation reports, plus plots.
- `run.py`, `generate.py`, `evaluate.py`, `requester.py`, `utils.py`, `plot.py` - scripts that implement generation, evaluation, utilities, and plotting.
- `run.ipynb`, `compare.ipynb` - exploratory notebooks for running and comparing outputs.

## Quick summary

- Generation: use a Requester (e.g. `OzwellRequester` or `OllamaRequester`) with a generation prompt (starts with `g`) to produce notes from `conversation` text in the dataset. Generated notes are written to `{model}/{prompt}/{idx}/gen_note.txt` (utilities handle versioning).
- Evaluation: similarity prompts (starts with `s`) are used to ask an LLM to compare the generated note and the reference note; ROUGE scores are computed via `rouge` package. Evaluation results are saved as `eval_report.json` alongside generated notes. Aggregate results are stored in `expiriments/`.

## Requirements

- Python 3.10+ recommended
- Core libraries used in the repo (install with pip):

```bash
pip install pandas python-dotenv requests ollama-client tqdm rouge matplotlib
```

Note: package names may vary for some third-party clients (`ollama` client). If you have a `requirements.txt` you prefer, use that.

## Environment variables

Place your API keys and other secrets in a `.env` file at the repo root (this repo already expects these variables):

- `OZWELL_SECRET_KEY` - secret key for Ozwell API
- `OZWELL_PUBLIC_KEY` - public key (optional)
- `OZWELL_WORKSPACE_ID`, `OZWELL_USER_ID` - optional workplace identifiers
- `OPENAI_SECRET_KEY` - (optional) OpenAI key if using OpenAI-backed requesters
- `OLLAMA_API_KEY` - (optional) API key for Ollama cloud models

Important: Do not commit real secret values to public repositories. Add `.env` to `.gitignore` before publishing.

## Usage

1) Generate notes for a set of samples

The helper functions in `generate.py` let you generate notes for rows in a DataFrame. The higher-level driver `run.py` shows a full flow and orchestration between generation and evaluation.

Example: generate n versions of notes for a set of sample ids with Ozwell:

```bash
python run.py
```

`run.py` will:

- load the dataset `augmented-clinical-notes/augmented_notes_30K.jsonl`
- select a list of sample IDs (inside the script)
- create a `Requester` using prompt generation prompt, and call `generate_n` to generate notes
- create requester using a similarity prompt, call `get_eval_df` and `get_rouge_avgs_df` (saves results in `expiriments/{generation_model_name}/{generation_prompt}`)

2) Generate programmatically (scripted)

You can use the Requester classes directly from `requester.py` in your scripts or REPL. Example (pseudo):

```python
from requester import OzwellRequester
req = OzwellRequester('g1')
note = req.send(conversation_text)
```

3) Evaluation

- `evaluate.py` contains an `Evaluator` class which loads any standard notes (symlinked in `standards/` as `standard_note_{idx}.txt`), combines them with the dataset references, and offers `get_rouge()` and `ai_eval()` to run ROUGE and model-based evaluations respectively. `evaluate.py` also contains the functions `get_eval_df`, which formats the ROUGE results and model-based evaluations into a dataframe for each row of the `gen_notes` dataframe passed as a parameter. Similarly, the `get_rouge_avgs_df` function in `evaluate.py` formats the ROUGE avgs. for each unique `idx` in the recieved `gen_notes` dataframe.
- Aggregated experiment outputs are written to `expiriments/{generation_model_name}/{generation_prompt_name}/eval_report{v}.json` and averages to `expiriments/{generation_model_name}/{generation_prompt_name}/rouge_avgs{v}.json` (see `run.py`).

4) Plotting

- `plot.py` reads `expiriments/{generation_model_name}/{generation_prompt_name}/eval_report.json` or `expiriments/{generation_model_name}/{generation_prompt_name}/rouge_avgs.json` (or other aggregated JSON) and writes scatter plots into `expiriments/{generation_model_name}/{generation_prompt_name}/plots/`. The plots are color-coded in accordance with the 'standard note' used to compare.

Files & conventions

- Prompts: `prompts/generation/g*.txt` (generation) and `prompts/similarity/s*.txt` (evaluation/similarity).
- Generated notes: `{model}/{prompt}/{idx}/gen_note.txt`. If multiple versions are generated, the utility appends a numeric suffix: `gen_note.txt`, `gen_note1.txt`, `gen_note2.txt`, etc.
- Evaluation report: `expiriments/{generation_model}/{generation_prompt}/eval_report.json` with keys like `{"rouge-1": ..., "ozwell-s1": ...}`.
- Standards: place symlinks to canonical standard notes in `standards/` named `standard_note_{idx}.txt` (use `utils.set_standard()` to create symlinks programmatically).
- Expiriments/old_results: contains older versions of `eval_reports.json` and `rouge_avgs.json` that are poorly formatted

## Developer notes

- `requester.py` provides two concrete Requester implementations:
  - `OzwellRequester(prompt_name, root_dir='./')` — uses the Ozwell API via HTTP
  - `OllamaRequester(model_name, prompt_name, root_dir='./')` — uses the Ollama client (local or cloud)
- `utils.py` contains helper functions for file matching, versioned writing, reading generated notes and building lists of gen note paths.
- `generate.py` provides `generate_loop` and `generate_n` for per-row generation into the `{model}/{prompt}/{idx}` structure.
- `evaluate.py` contains evaluation helper methods and is used by `run.py`.

## Quick troubleshooting

- If prompts are not found, confirm prompt file names in `prompts/generation/` and `prompts/similarity/` and that `Requester.set_prompt()` uses the correct prompt name (no `.txt`).
- If API calls fail, check your `.env` credentials and network access. The repo expects keys in environment variables (loaded via `python-dotenv`).
- If ROUGE errors occur, ensure the `rouge` package is installed and matches the API used in the code.

## Examples & results

- Example generated note: `ozwell/g1/224/gen_note.txt` (a sample clinical note generated by Ozwell) is present in the repo to inspect formatting and style.
- Aggregated ROUGE results and plots are available under `expiriments/ozwell/g1/`.

## License & acknowledgments

- Dataset / prompt credit: The generation prompt `prompts/g1.txt` is adapted from the Medinote project: https://github.com/EPFL-IC-Make-Team/medinote
