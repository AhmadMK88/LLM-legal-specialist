import json
import re

def extract_json_block(text):
    candidates = re.findall(r"\{[\s\S]*?\}", text)

    if not candidates:
        raise ValueError("No JSON found in model output\nOutput was:\n" + text)

    for block in reversed(candidates):
        try:
            cleaned = re.sub(r"//.*", "", block)   # remove comments
            cleaned = cleaned.replace("\n", "")
            cleaned = cleaned.replace("\t", "")
            cleaned = cleaned.replace(",}", "}")   # fix trailing commas

            json.loads(cleaned)  # test
            return cleaned
        except:
            continue

    raise ValueError("No valid JSON block found after cleaning.\nCandidates:\n" + str(candidates))

def parse_json(json_str):
    match = re.search(r"\{.*\}", json_str, flags=re.DOTALL)
    if not match:
        raise ValueError("No valid JSON object found.")

    cleaned = match.group(0).strip()
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decode failed after cleaning: {e}\nCleaned JSON:\n{cleaned}")

