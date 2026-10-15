"""Generate archival tables and optional scientific plots from validated results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from trackmaniarl.research.manifest import write_json


def _tex(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "^": r"\^{}",
        "~": r"\~{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def write_report(result: dict[str, Any], output: Path, *, plot: bool = False) -> None:
    """Create JSON, CSV, LaTeX, and optionally PDF/PNG plots, without overwriting files."""
    names = ["results.json", "runs.csv", "trials.csv", "table.tex"]
    if plot:
        names += ["finish-rate.pdf", "finish-rate.png"]
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    if any((output / name).exists() for name in names):
        raise ValueError("report output already exists; use a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "results.json", result)
    for name, rows in (("runs.csv", result["runs"]), ("trials.csv", result["trials"])):
        with (output / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"Condition & Trials & Finish rate & Cost & Median (s) \\",
        r"\hline",
    ]
    for row in result["runs"]:
        median = "--" if row["median_finish_s"] is None else f"{row['median_finish_s']:.2f}"
        lines.append(
            f"{_tex(row['condition'])} & {row['trials']} & {row['finish_rate']:.3f} & "
            f"{row['normalized_cost']:.3f} & {median} " + r"\\"
        )
    lines.append(r"\end{tabular}")
    (output / "table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if plot:
        rows = result["runs"]
        rates = [r["finish_rate"] for r in rows]
        errors = [
            [r["finish_rate"] - r["finish_rate_trial_wilson95"][0] for r in rows],
            [r["finish_rate_trial_wilson95"][1] - r["finish_rate"] for r in rows],
        ]
        figure, axis = plt.subplots(figsize=(7, max(2.5, len(rows) * 0.55)))
        axis.errorbar(rates, list(range(len(rows))), xerr=errors, fmt="o", capsize=3)
        axis.set_yticks(list(range(len(rows))), [r["run_id"] for r in rows])
        axis.set(
            xlim=(0, 1.02),
            xlabel="Finish rate; 95% Wilson interval within checkpoint",
            title=f"{result['evidence'].title()} evidence: trials are not training seeds",
        )
        figure.tight_layout()
        figure.savefig(output / "finish-rate.pdf")
        figure.savefig(output / "finish-rate.png", dpi=180)
        plt.close(figure)
