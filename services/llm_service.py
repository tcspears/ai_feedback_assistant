from typing import Dict, List
from openai import OpenAI
from anthropic import Anthropic
import logging

class LLMService:
    def __init__(self, openai_client: OpenAI, anthropic_client: Anthropic):
        self.openai_client = openai_client
        self.anthropic_client = anthropic_client
        
        # Model mapping to handle legacy model names
        self.model_mapping = {
            # Legacy models mapped to current models
            'gpt-4': 'gpt-4o',
            'gpt-4-mini': 'gpt-4o-mini',
            'gpt-3.5-turbo': 'gpt-4o-mini',  # Map older models to newer ones
            'claude-3-opus': 'claude-3-5-sonnet-latest',
            'claude-3-sonnet': 'claude-3-5-sonnet-latest',
            'claude-3-haiku': 'claude-3-5-haiku-latest',
        }
        
        # Supported model handlers
        self.model_handlers = {
            'gpt-4o': self._handle_openai,
            'gpt-4o-mini': self._handle_openai,
            'o1-preview': self._handle_openai,
            'o1-mini': self._handle_openai,
            'claude-3-5-haiku-latest': self._handle_anthropic,
            'claude-3-5-sonnet-latest': self._handle_anthropic,
            # Add more models as needed
        }
    
    def generate_response(self, model: str, messages: List[Dict[str, str]], system_msg: str = None) -> str:
        """
        Generate a response using the specified model.
        
        Args:
            model: The model identifier (e.g., 'gpt-4', 'claude-3-opus')
            messages: List of message dictionaries with 'role' and 'content'
            system_msg: Optional system message
            
        Returns:
            Generated response text
        """
        # Map legacy model names to supported models
        if model in self.model_mapping:
            logging.info(f"Mapping legacy model '{model}' to '{self.model_mapping[model]}'")
            model = self.model_mapping[model]
        
        handler = self.model_handlers.get(model)
        if not handler:
            supported_models = list(self.model_handlers.keys()) + list(self.model_mapping.keys())
            raise ValueError(f"Unsupported model: {model}. Supported models are: {', '.join(supported_models)}")
        
        return handler(model, messages, system_msg)
    
    def _handle_openai(self, model: str, messages: list, system_msg: str = None) -> str:
        if system_msg:
            messages = [{"role": "system", "content": system_msg}] + messages
            
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content.strip()
    
    def _handle_anthropic(self, model: str, messages: list, system_msg: str = None) -> str:
        response = self.anthropic_client.messages.create(
            model=model,
            system=system_msg if system_msg else "",
            messages=messages,
            max_tokens=1000
        )
        return response.content[0].text.strip() 