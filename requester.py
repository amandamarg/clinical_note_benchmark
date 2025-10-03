import os
import requests
from abc import ABC, abstractmethod
from ollama import Client
import re

class Requester(ABC):
    def __init__(self, model_name, prompt_name, api_key, root_dir='./'):
        self.root_dir = root_dir
        self.model_name = model_name
        self.set_prompt(prompt_name)
        self.header = self.build_header(api_key)
    
    @abstractmethod
    def build_header(self, api_key):
        pass

    def set_prompt(self, prompt_name):
        prompt_type = prompt_name[0].lower()
        assert prompt_type in ['s', 'g']
        mode = "similarity" if  prompt_type == 's' else "generation"
        prompt_path = os.path.join(self.root_dir, "prompts", mode, prompt_name + ".txt")
        assert os.path.exists(prompt_path)
        self.prompt_name = prompt_name
        with open(prompt_path, 'r') as file:
            self.prompt = file.read()

    @abstractmethod
    def send(self, data):
        pass

class OzwellRequester(Requester):
    def __init__(self, prompt_name, api_key, root_dir='./'):
        super().__init__('ozwell', prompt_name, api_key, root_dir)
    
    def build_header(self, api_key):
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def send(self, data):
        url = 'https://ai.bluehive.com/api/v1/completion'
        payload = {"prompt": self.prompt, "systemMessage": data}
        resp = requests.post(url, headers=self.header, json=payload)
        resp = resp.json()
        try:
            content = resp['choices'][0]['message']['content']
        except:
            status_code = resp.status_code
            print(f'Response status: {status_code}')
        return content


class OllamaRequester(Requester):
    def __init__(self, model_name, prompt_name, api_key=None, root_dir='./'):
        super().__init__(model_name, prompt_name, api_key, root_dir)
        if re.search(r'-cloud$', self.model_name):
            assert api_key is not None
            self.client = Client(
                host='https://ollama.com',
                headers=self.header
            )
        else:
            self.client = Client(
                host='http://localhost:11434',
                headers=self.header
            )

    def build_header(self, api_key):
        if api_key:
            return {"Authorization": f"{api_key}", "Content-Type": "application/json"}
        else:
            return {"Content-Type": "application/json"}

    def send(self, data):
        response = self.client.generate(model=self.model_name, system=self.prompt, prompt=data, stream=False)
        return response.response


