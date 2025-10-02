prompts:
    *prompts used to generate notes are found under prompts/generate and are named starting with a g (ex. g1.txt)
    *prompts used to assess similarity between notes are found under prompts/similarity and are named starting with a s (ex. s1.txt)

generated notes:
    *generated notes are stored using the file name gen_note.txt under {model}/{prompt}/{idx} where model is the model used to generate that note, prompt is the name of the file (w/o the extension) used to generate the note, and {idx} refers to the idx corresponding to the idx of the transcript used to generate that note

evals:
    *the evaluation report will be named eval_report.json and can be found under {gen_model}/{gen_prompt}/{idx} where gen_model is the model that generated the notes being evaluated, gen_prompt is the prompt that generated the notes being evaluated, and {idx} is the idx of the transcript used to generate the note that report is evaluating
    *the eval report will contain fields formatted as "{eval_model}": {"{similarity_prompt}": {response}} the {response} is the response of {eval_model} on the similarity between the reference note and the generated note when prompted with {similarity_prompt} 



Citations and Acknowledgments:
prompts/g1.txt used from https://github.com/EPFL-IC-Make-Team/medinote