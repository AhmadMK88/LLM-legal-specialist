import json
import re
from typing import Dict


_JSON_OBJECT_REGEX = re.compile(r"\{[\s\S]*?\}")
_TRAILING_COMMA_REGEX = re.compile(r",\s*([}\]])")
_COMMENT_REGEX = re.compile(r"//.*?$", flags=re.MULTILINE)


def _clean_json(text: str) -> str:
    """
    Cleans common LLM JSON issues:
    - Removes comments
    - Removes trailing commas
    - Strips unnecessary whitespace

    Args: 
        - text(str): json block to be cleaned
    
    Returns:
        - test(str): cleaned json block
    """
    text = _COMMENT_REGEX.sub("", text)
    text = _TRAILING_COMMA_REGEX.sub(r"\1", text)
    return text.strip()


def extract_json_block(text: str) -> Dict:
    """
    Extracts the LAST valid JSON object from a text blob.
    LLMs often emit multiple JSON-like blocks; the last one
    is usually the final answer.

    Args:
        - text(str): text to extract json block from
    
    Returns:
        - json_block(Dict): extracted json block
    """
    matches = _JSON_OBJECT_REGEX.findall(text)

    if not matches:
        raise ValueError(
            "No JSON object found in model output.\n"
            f"Raw output:\n{text}"
        )

    for candidate in reversed(matches):
        cleaned = _clean_json(candidate)
        try:
            json_block = json.loads(cleaned)
            return json_block
        except json.JSONDecodeError:
            continue

    raise ValueError(
        "JSON blocks were found, but none were valid after cleaning.\n"
        f"Candidates:\n{matches}"
    )
