import pytest
from utils.prompt_builder import StructuredPrompt

def test_add_simple_section():
    prompt = StructuredPrompt()
    prompt.add_section("test", "content")
    assert prompt.sections["test"] == "content"
    assert prompt.section_order == ["test"]

def test_add_section_with_subsections():
    prompt = StructuredPrompt()
    subsections = {
        "sub1": "content1",
        "sub2": "content2"
    }
    prompt.add_section("test", "main content", subsections)
    expected = "<sub1>content1</sub1>\n<sub2>content2</sub2>"
    assert prompt.sections["test"] == expected

def test_build_single_section():
    prompt = StructuredPrompt()
    prompt.add_section("test", "content")
    result = prompt.build()
    assert result == "<test>content</test>"

def test_build_multiple_sections():
    prompt = StructuredPrompt()
    prompt.add_section("section1", "content1")
    prompt.add_section("section2", "content2")
    result = prompt.build()
    expected = "<section1>content1</section1>\n\n<section2>content2</section2>"
    assert result == expected

def test_build_maintains_order():
    prompt = StructuredPrompt()
    # Add sections in different order than desired
    prompt.add_section("section2", "content2")
    prompt.add_section("section1", "content1")
    # But they should appear in order of addition
    result = prompt.build()
    expected = "<section2>content2</section2>\n\n<section1>content1</section1>"
    assert result == expected

def test_empty_prompt():
    prompt = StructuredPrompt()
    assert prompt.build() == ""

def test_section_with_empty_content():
    prompt = StructuredPrompt()
    prompt.add_section("test", "")
    assert prompt.build() == "<test></test>"

def test_complex_nested_structure():
    prompt = StructuredPrompt()
    prompt.add_section("context", "You are an AI assistant")
    prompt.add_section("instructions", "Please analyze", {
        "format": "Use markdown",
        "style": "Be concise",
        "tone": "Professional"
    })
    result = prompt.build()
    expected = (
        "<context>You are an AI assistant</context>\n\n"
        "<instructions><format>Use markdown</format>\n"
        "<style>Be concise</style>\n"
        "<tone>Professional</tone></instructions>"
    )
    assert result == expected

def test_duplicate_section_names():
    prompt = StructuredPrompt()
    prompt.add_section("test", "content1")
    prompt.add_section("test", "content2")
    # Should update content but not add duplicate to order
    assert prompt.sections["test"] == "content2"
    assert prompt.section_order.count("test") == 1
    assert prompt.build() == "<test>content2</test>"