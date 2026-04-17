import re


def normalise_llm_output(raw: str) -> str:
    """
    Strip chain-of-thought blocks, markdown fences, and excess whitespace.

    Processing order:
    1. Strip <think>...</think> block (keeps content after </think>)
    2. Strip loose </think> tag with no opening block
    3. Strip ```json or ``` fences
    4. Strip leading/trailing whitespace
    """
    if '<think>' in raw and '</think>' in raw:
        raw = raw.split('</think>', 1)[-1]
    elif '</think>' in raw:
        raw = raw.replace('</think>', '')

    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)

    return raw.strip()
