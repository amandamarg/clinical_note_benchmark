import requests
import os
from abc import ABC, abstractmethod

class Generator(ABC):
    def __init__(self, model_name, prompt_name, api_key, root_dir = "./"):
        self.root_dir = root_dir
        self.model_name = model_name
        self.set_prompt(prompt_name)
        self.header = self.build_header(api_key)
    
    @abstractmethod
    def build_header(self, api_key):
        pass
    
    def set_prompt(self, prompt_name):
        prompt_path = os.path.join(self.root_dir, "prompts", prompt_name + ".txt")
        assert os.path.exists(prompt_path)
        with open(prompt_path, 'r') as file:
            self.prompt = file.read()
        self.output_dir = os.path.join(self.root_dir, self.model_name, prompt_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @abstractmethod
    def generate_note(self, transcript, idx):
        pass

    def write_note(self, idx, content):
        file_path = os.path.join(self.output_dir, f'transcript_{idx}_note.txt')
        with open(file_path, 'w') as file:
            file.write(content)


class OzwellGenerator(Generator):
    def __init__(self, prompt_name, api_key, root_dir = "./"):
        super().__init__("ozwell", prompt_name, api_key, root_dir)
    
    def build_header(self, api_key):
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    def generate_note(self, transcript, idx):
        payload = {"prompt": self.prompt, "systemMessage": transcript}
        url = 'https://ai.bluehive.com/api/v1/completion'
        resp = requests.post(url, headers=self.header, json=payload)
        try:
            content = resp.json()['choices'][0]['message']['content']
            self.write_note(idx, content)
        except:
            print(f"Response {resp.status_code}, idx: {idx}, resp: {resp.json()}")