"""
LLM Provider implementations for macLLM.
Supports OpenAI and Ollama backends.
"""

import abc
import base64
import json
import httpx
import openai
import requests


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""
    
    @abc.abstractmethod
    def generate(self, text):
        """Generate text response from prompt."""
        pass
        
    @abc.abstractmethod
    def generate_with_image(self, text, image_path):
        """Generate text response from prompt and image."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, api_key, model="gpt-4", temperature=0.0):
        """Initialize OpenAI provider with API key and parameters."""
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=api_key)
        
    def generate(self, text):
        """Generate text using OpenAI chat completion API."""
        c = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": str(text)},
            ],
            temperature=self.temperature,
        )
        return c.choices[0].message.content
        
    def encode_image(self, image_path):
        """Encode image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    def generate_with_image(self, text, image_path):
        """Generate text from prompt and image using OpenAI vision API."""
        base64_image = self.encode_image(image_path)
        if base64_image is None:
            print(f'Image encoding failed.')
            return None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{text}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                generated_text = response_data['choices'][0]['message']['content']
                print(f'Generated Text: {generated_text}')
                return generated_text
            else:
                print('No generated content found.')
                return None
        else:
            print(f'Failed to generate content. Status Code: {response.status_code}')
            print(response.json())
            return None


class OllamaProvider(LLMProvider):
    """Ollama local API provider implementation."""
    
    def __init__(self, model="llama2", temperature=0.0):
        """Initialize Ollama provider with model and parameters."""
        self.model = model
        self.temperature = temperature
        self.base_url = "http://localhost:11434/api"
        
    def generate(self, text):
        """Generate text using Ollama local API."""
        payload = {
            "model": self.model,
            "prompt": str(text),
            "temperature": self.temperature,
            "stream": False
        }
        
        try:
            response = httpx.post(f"{self.base_url}/generate", json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except httpx.HTTPError as e:
            print(f"Error calling Ollama API: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected response format from Ollama: {e}")
            return None
            
    def generate_with_image(self, text, image_path):
        """Generate text from prompt and image using Ollama API.
        
        Note: Currently raises NotImplementedError as Ollama doesn't support
        image input in the same way as OpenAI. This may change in future versions.
        """
        raise NotImplementedError(
            "Image generation not supported with Ollama provider. "
            "Use OpenAI provider for image-based generation."
        )