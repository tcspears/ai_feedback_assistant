import yaml
from typing import Dict, Any, Tuple
from utils.prompt_builder import StructuredPrompt

class PromptLoader:
    def __init__(self, yaml_file: str = 'prompts.yaml'):
        with open(yaml_file, 'r') as file:
            self.prompts = yaml.safe_load(file)
    
    def create_prompt(self, prompt_name: str, **kwargs) -> Tuple[StructuredPrompt, str]:
        """
        Create a StructuredPrompt from a template, filling in dynamic values.
        
        Args:
            prompt_name: Name of the prompt template to use
            **kwargs: Dynamic values to fill into the template
        
        Returns:
            Tuple of (StructuredPrompt object, system message)
        """
        if prompt_name not in self.prompts:
            raise ValueError(f"Unknown prompt template: {prompt_name}")
            
        template = self.prompts[prompt_name]
        prompt = StructuredPrompt()
        
        # Get system message
        system_msg = template.get('system_msg', '')
        
        # Process sections
        for section_name, section_data in template['sections'].items():
            content = section_data.get('content')
            subsections = section_data.get('subsections')
            
            # Handle dynamic content
            if content is not None:
                content = content.format(**kwargs) if kwargs else content
            elif section_name in kwargs:
                content = kwargs[section_name]
                
            # Handle dynamic subsections
            if subsections:
                filled_subsections = {}
                for sub_name, sub_content in subsections.items():
                    if sub_content is not None:
                        filled_subsections[sub_name] = sub_content.format(**kwargs) if kwargs else sub_content
                    elif sub_name in kwargs:
                        filled_subsections[sub_name] = kwargs[sub_name]
                subsections = filled_subsections
            
            if content is not None or subsections:
                prompt.add_section(section_name, content or "", subsections)
        
        return prompt, system_msg 