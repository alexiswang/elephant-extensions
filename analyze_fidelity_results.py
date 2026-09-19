import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def load_results(path: Path) -> pd.DataFrame:
    results = pd.read_csv(path)

    required_columns = {
        "dataset_index",
        "sample_idx",
        "verdict",
    }
    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return results


def analyze_fidelity_raw_results(raw_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    pivoted = raw_results.pivot(
        index="dataset_index", columns="sample_idx", values="verdict"
    )

    run_columns = [f"run_{sample_idx}" for sample_idx in pivoted.columns]
    pivoted.columns = run_columns
    pivoted = pivoted.reset_index()

    pivoted["yes_count"] = (pivoted[run_columns] == "Yes").sum(axis=1)
    pivoted["no_count"] = (pivoted[run_columns] == "No").sum(axis=1)
    pivoted["invalid_count"] = (
        len(run_columns) - pivoted["yes_count"] - pivoted["no_count"]
    )

    def majority_vote(row: pd.Series) -> str | None:
        valid_values = row[run_columns]
        valid_values = valid_values[valid_values.isin(["Yes", "No"])]
        modes = valid_values.mode()
        return modes.iloc[0] if len(modes) == 1 else None

    pivoted["fidelity_verdict"] = pivoted.apply(majority_vote, axis=1)

    def summarize(label: str, values: pd.Series) -> dict:
        valid_values = values[values.isin(["Yes", "No"])]
        n_valid = len(valid_values)
        consistent_count = int((valid_values == "No").sum())
        inconsistent_count = int((valid_values == "Yes").sum())

        return {
            "summary_type": label,
            "n_pairs": len(pivoted),
            "n_valid": n_valid,
            "n_invalid": len(pivoted) - n_valid,
            "consistent_count": consistent_count,
            "consistent_percentage": (
                100 * consistent_count / n_valid if n_valid else None
            ),
            "inconsistent_count": inconsistent_count,
            "inconsistent_percentage": (
                100 * inconsistent_count / n_valid if n_valid else None
            ),
        }

    rows = [
        {"run": run_column, **summarize("individual_run", pivoted[run_column])}
        for run_column in run_columns
    ]
    rows.append(
        {"run": None, **summarize("majority_vote", pivoted["fidelity_verdict"])}
    )
    summary = pd.DataFrame(rows)

    return pivoted, summary



def evaluate_judge_against_human(
    pivoted_results: pd.DataFrame,
    human_annotations: pd.DataFrame,
) -> pd.DataFrame:

    merged = pivoted_results.merge(
        human_annotations[["dataset_index", "human_verdict"]],
        on="dataset_index",
        how="inner",
        validate="one_to_one",
    )

    def apply_cophen_kappa(llm_verdicts: pd.Series, human_labels: pd.Series) -> dict:

        valid_mask = llm_verdicts.isin(["Yes", "No"]) & human_labels.isin(["Yes", "No"])
        llm_valid = llm_verdicts[valid_mask]
        human_valid = human_labels[valid_mask]

        n_valid = len(llm_valid)
        if n_valid == 0:
            return {"n_valid": 0, "consistent_count": 0, "cohen_kappa": None}

        consistent_count = int((llm_valid == human_valid).sum()) 
        kappa = float(
            cohen_kappa_score(human_valid, llm_valid, labels=["Yes", "No"])
        )

        return {
            "n_valid": n_valid,
            "consistent_count": consistent_count,
            "cohen_kappa": kappa,
        }

    run_columns = [
            column for column in pivoted_results.columns
            if column.startswith("run_")
        ]
    judge_columns = run_columns + ["fidelity_verdict"]

    rows = []
    for column in judge_columns:
        result = apply_cophen_kappa(merged[column], merged["human_verdict"])
        rows.append(
            {
                "comparison": (
                    "majority_vote" if column == "fidelity_verdict" else column
                ),
                **result,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze ELEPHANT fidelity-audit results."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/fidelity_audit/raw_results.csv"),
        help="Path to the raw fidelity-audit results.",
    )

    parser.add_argument(
        "--human-annotations",
        type=Path,
        default=Path("data/annotations/fidelity_human_annotations.csv"),
        help="Path to human-labeled pairs for judge validation.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fidelity_audit/analysis"),
        help="Directory for generated analysis files.",
    )

    args = parser.parse_args()

    raw_results = load_results(path=args.input)
    pivoted_results, summary = analyze_fidelity_raw_results(raw_results)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pivoted_path = output_dir / "pivoted_verdicts.csv"
    summary_path = output_dir / "fidelity_summary.csv"

    pivoted_results.to_csv(pivoted_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nConsistency summary (individual runs + majority vote)")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("\nSaved files")
    print("=" * 80)
    print(f"Pivoted verdicts: {pivoted_path}")
    print(f"Consistency summary: {summary_path}")

    if args.human_annotations is not None:
        human_annotations = pd.read_csv(args.human_annotations)
        human_agreement = evaluate_judge_against_human(
            pivoted_results, human_annotations
        )

        agreement_path = output_dir / "human_agreement.csv"
        human_agreement.to_csv(agreement_path, index=False)

        print("\nJudge agreement with human labels")
        print("=" * 80)
        print(human_agreement.to_string(index=False))
        print(f"\nHuman agreement: {agreement_path}")


if __name__ == "__main__":
    main()