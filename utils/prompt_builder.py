from typing import Dict, Optional
from dataclasses import dataclass, field

@dataclass
class StructuredPrompt:
    sections: Dict[str, str] = field(default_factory=dict)
    section_order: list[str] = field(default_factory=list)  # Empty by default

    def add_section(self, name: str, content: str, 
                   subsections: Optional[Dict[str, str]] = None) -> None:
        if subsections:
            # Handle nested sections
            content = "\n".join([
                f"<{sub_name}>{sub_content}</{sub_name}>"
                for sub_name, sub_content in subsections.items()
            ])
        self.sections[name] = content
        # Add to section_order if not already present
        if name not in self.section_order:
            self.section_order.append(name)

    def build(self) -> str:
        # Build prompt maintaining specified order
        ordered_sections = []
        for section in self.section_order:
            if section in self.sections:
                ordered_sections.append(
                    f"<{section}>{self.sections[section]}</{section}>"
                )
        return "\n\n".join(ordered_sections) 