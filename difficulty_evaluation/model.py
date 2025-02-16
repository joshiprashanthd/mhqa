from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from groq import Groq
from dotenv import load_dotenv
import os
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_result
)

load_dotenv()

class BaseModel:
    def __init__(self, model_name, device):
        self.device = device
        self.model_name = model_name

    def generate_text(self, prompt, **kwargs):
        raise NotImplementedError
    
class OpenAIGPT(BaseModel):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        self.openai = OpenAI()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(8)) 
    def generate_text(self, prompt, **kwargs):
        response = self.openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            stream=False,
            temperature=0.7,
        )
        generated_text = response.choices[0].message.content
        print(generated_text)
        return generated_text
    

class Llama(BaseModel):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
    
    def generate_text(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=1024)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
class GroqModel(BaseModel):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    def generate_text(self, prompt, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content

class HFApiModel(BaseModel):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.api_key = os.getenv("HF_API_KEY")
    
    def _try(self, api_key, prompt):
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(
            self.api_url,
            headers=headers,
            json={
                "inputs": prompt,
                "options": {"use_cache": True},
                "parameters": {"max_new_tokens": 512, "temperature": 0.7, "return_full_text": False}
            }
        )

        print(response.content)

        if response.status_code == 200:
            response_data = response.json()
            if isinstance(response_data, list) and 'generated_text' in response_data[0]:
                return response_data[0]['generated_text'].strip()
        return None
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(8), retry=retry_if_result(lambda x: x is None)) 
    def generate_text(self, prompt, **kwargs):
        return self._try(self.api_key, prompt)
