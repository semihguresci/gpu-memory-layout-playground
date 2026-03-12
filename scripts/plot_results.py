from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "results" / "tables" / "benchmark_results.csv"
OUTPUT_PNG = ROOT / "results" / "charts" / "benchmark_summary.png"


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["experiment"], df["average_ms"])
    ax.set_title("Benchmark Average Time")
    ax.set_ylabel("Average Time (ms)")
    ax.set_xlabel("Experiment")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)


if __name__ == "__main__":
    main()
