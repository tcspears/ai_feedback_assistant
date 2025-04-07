from typing import Dict, List
from openai import OpenAI
from anthropic import Anthropic
import logging
import time
import random
import anthropic

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
        # Add retry logic with exponential backoff
        max_retries = 5
        retry_count = 0
        base_delay = 2  # starting delay in seconds
        
        while retry_count < max_retries:
            try:
                response = self.anthropic_client.messages.create(
                    model=model,
                    system=system_msg if system_msg else "",
                    messages=messages,
                    max_tokens=1000
                )
                return response.content[0].text.strip()
                
            except anthropic.InternalServerError as e:
                # Check if it's the 529 Overloaded error specifically
                error_type = getattr(e, 'type', None)
                error_status = getattr(e, 'status_code', None)
                
                if error_status == 529 or (hasattr(e, 'response') and 'overloaded_error' in str(e.response)):
                    retry_count += 1
                    if retry_count >= max_retries:
                        logging.error(f"Maximum retries reached for Anthropic API. Error: {e}")
                        raise ValueError(f"Anthropic API is currently overloaded. Please try again later or use a different model.") from e
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
                    logging.warning(f"Anthropic API overloaded. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
                else:
                    # If it's not an overloaded error, raise immediately
                    logging.error(f"Anthropic API error: {e}")
                    raise ValueError(f"Error with Anthropic API: {e}") from e
            
            except Exception as e:
                logging.error(f"Unexpected error with Anthropic API: {e}")
                raise ValueError(f"Error with Anthropic API: {e}") from e 