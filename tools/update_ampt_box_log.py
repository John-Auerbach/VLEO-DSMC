#!/usr/bin/env python3
"""Regenerate the ampt_box log table in README.md from data/ampt_box_log.tsv.

Reads the TSV (first row = header, remaining rows = data) and rewrites the
markdown table between the <!-- AMPT_BOX_LOG_START --> and
<!-- AMPT_BOX_LOG_END --> markers in README.md.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TSV = ROOT / "data" / "ampt_box_log.tsv"
README = ROOT / "README.md"
START = "<!-- AMPT_BOX_LOG_START -->"
END = "<!-- AMPT_BOX_LOG_END -->"


def build_table() -> str:
    rows = [line.rstrip("\n").split("\t") for line in TSV.read_text().splitlines() if line != ""]
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    ncols = len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * ncols) + "|",
    ]
    for r in body:
        r = (r + [""] * ncols)[:ncols]
        lines.append("| " + " | ".join(c if c else " " for c in r) + " |")
    return "\n".join(lines)


def main() -> None:
    text = README.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"Markers {START} / {END} not found in {README}")
    pre, rest = text.split(START, 1)
    _, post = rest.split(END, 1)
    new = f"{pre}{START}\n{build_table()}\n{END}{post}"
    README.write_text(new)
    print(f"Updated table in {README} from {TSV}")


if __name__ == "__main__":
    main()
