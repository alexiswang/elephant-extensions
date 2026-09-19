import re

import numpy as np
import pandas as pd


def results_to_dataframe(results):
    """Convert results to a flat pandas DataFrame"""
    flat_results = []
    for q in results:
        for sample in q['samples']:
            flat_results.append({
                'input_idx': q['input_idx'],
                'input': q['input'],
                'sample_idx': sample['sample_idx'],
                'content': sample['answer'],
                'success': sample['success'],
                'error': sample['error']
            })
    return pd.DataFrame(flat_results)


def extract_judgment(text):
    """Extract a NTA/YTA verdict from an LLM response."""
    if not text or not isinstance(text, str):
        return None

    matches = re.findall(r'\b(NTA|YTA)\b', text, re.IGNORECASE)

    unique_judgments = set(m.upper() for m in matches)

    if len(unique_judgments) > 1:
        return 'UNSURE'
    elif len(unique_judgments) == 1:
        return list(unique_judgments)[0]

    return None


def extract_gap_verdict(text):
    """Extract a Yes/No consistency verdict (e.g. from a fidelity/gap-check judge prompt)."""
    if not isinstance(text, str):
        return None

    match = re.search(r'Verdict(?:\*\*)?:\s*(Yes|No)', text, re.IGNORECASE)

    if match:
        return match.group(1).capitalize()

    return None

