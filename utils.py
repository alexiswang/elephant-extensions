import re

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest


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


def compare_groups(n1, n2, counts_df, cols=['temp_1', 'temp_2'], alternative='two-sided', alpha=0.05):
    """
    Compare proportions between two groups across multiple metrics.

    Args:
        n1: total observations in group 1
        n2: total observations in group 2
        counts_df: DataFrame with a 'Metric' column and one count column per group
        cols: the two count columns in counts_df to compare, e.g. ['group1_col', 'group2_col']
        alternative: 'two-sided', 'larger' (group1 > group2), or 'smaller'
        alpha: significance level

    Returns:
        dict of results per metric
    """
    results = {}

    for _, row in counts_df.iterrows():
        name = row['Metric']
        count = [row[cols[0]], row[cols[1]]]
        nobs = [n1, n2]

        z_stat, p_value = proportions_ztest(count, nobs, alternative=alternative)

        p1 = count[0] / n1
        p2 = count[1] / n2

        results[name] = {
            'p1': p1,
            'p2': p2,
            'diff': p1 - p2,
            'z': z_stat,
            'p_value': p_value,
            'significant': p_value < alpha
        }

        print(f"\n{name}:")
        print(f"  Group 1: {p1:.3f} ({count[0]}/{n1})")
        print(f"  Group 2: {p2:.3f} ({count[1]}/{n2})")
        print(f"  Diff: {p1 - p2:+.3f}")
        print(f"  z = {z_stat:.3f}, p = {p_value:.4f}")
        print(f"  {'Significant' if p_value < alpha else 'Not significant'} at α={alpha}")

    return results
