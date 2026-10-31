"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll past it or close the window. This module writes the full report (accuracy, classification report, fairness table) to a timestamped file in `results/` instead, so youcan open it again later, or compare two runs side by side after changing something in config.yaml.
"""

import os
import pandas as pd
from datetime import datetime


def save_run(
    results_dir: str,
    run: str,
    config: dict,
    report_text: str
) -> str:
    """
    Saves the complete report in a timestamped text file.
    The run identifier is provided by main.py.
    """

    os.makedirs(results_dir, exist_ok=True)

    path = os.path.join(
        results_dir,
        f"run_{run}.txt"
    )

    header = (
        f"Run: {run}\n"
        f"Model: {config['model']['type']} "
        f"params={config['model'].get('params', {})}\n"
        f"Selection metric: "
        f"{config['cv']['selection_metric']}\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)

    return path





def save_scores(path: str, row: dict) -> pd.DataFrame:
    """
    Appends one run to the CSV at path
    (created if needed) and returns all runs.
    """

    runs = (
        pd.read_csv(path, dtype={"run": str})
        if os.path.exists(path)
        else pd.DataFrame()
    )

    runs = pd.concat(
        [runs, pd.DataFrame([row])],
        ignore_index=True
    )

    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True
    )

    runs.to_csv(path, index=False)

    return runs

