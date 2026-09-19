# ELEPHANT Benchmark Extensions

This repository contains the code and analysis for a BlueDot Impact Technical AI Safety project examining the ELEPHANT benchmark for measuring social and moral sycophancy in language models.

The project investigates three research questions:

- **RQ1:** Do reasoning models exhibit different levels of moral sycophancy?
- **RQ2:** To what extent does information loss during the perspective-flipping process compromise benchmark validity?
- **RQ3:** Does the measured sycophancy rate capture a single coherent phenomenon?

The experiments extend the ELEPHANT benchmark with:

- A fidelity audit of the AITA-NTA-FLIP dataset
- A comparison of reasoning and non-reasoning models
- Prompt-nudge ablations measuring sensitivity to user opinions
- Analysis of model judgments on original and perspective-flipped posts

Detailed methods, results, limitations, and references are provided in [report.md](report.md).

## Repository Structure

- `data/` - Source and prepared datasets
- `run_fidelity_check.py` - Run the dataset fidelity audit
- `analyze_fidelity_results.py` - Analyze fidelity-audit outputs
- `run_experiments.py` - Run model evaluations across prompt conditions
- `analyze_experiment_results.py` - Analyze model-evaluation outputs
- `notebooks/` - Exploratory analysis and qualitative inspection
- `report.md` - Full project report
- `results/` - Generated experiment outputs

## Reproducibility

The source dataset is included in `data/`. Model evaluations use the OpenRouter API and require an API key configured through the `OPENROUTER_API_KEY` environment variable.

Generated model outputs and result files are not included in version control by default. Because the code has been cleaned up since the original experiments, regenerated results may differ from those reported in `report.md`; the experimental procedure remains the same.

## License and Data

The source dataset is from the ELEPHANT benchmark and is included under its original licensing terms. Please consult the source paper and dataset documentation before redistributing it.