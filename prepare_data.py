from pathlib import Path

import pandas as pd


SOURCE_PATH = Path("data/elephant/AITA-NTA-FLIP_og_YTANTA.csv")
OUTPUT_PATH = Path("data/prepared/aita_nta_flip_sample.csv")

N_ROWS = 400

ORIGINAL_COLUMN = "original_NTA_post"
FLIPPED_COLUMN = "flipped_post"


def main() -> None:
    source = pd.read_csv(SOURCE_PATH, index_col=0)

    sample = source.iloc[:N_ROWS][[ORIGINAL_COLUMN, FLIPPED_COLUMN]].copy()
    sample.index.name = "dataset_index"
    sample = sample.reset_index()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(sample)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()