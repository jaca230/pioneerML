"""
Convert tutorial Python scripts (with ``# %%`` cells) into Jupyter notebooks.

Usage:
    python scripts/convert_tutorials_to_notebooks.py

By default this scans ``notebooks/tutorials`` for ``*.py`` files and writes
matching ``.ipynb`` notebooks alongside them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import nbformat as nbf


def parse_cells(lines: Iterable[str]):
    cells = []
    current: list[str] = []
    cell_type = "code"

    def flush():
        nonlocal current, cell_type
        if not current:
            return
        if cell_type == "markdown":
            md_lines = []
            for line in current:
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line.startswith("#"):
                    md_lines.append(line[1:].lstrip())
                else:
                    md_lines.append(line)
            cells.append(nbf.v4.new_markdown_cell("".join(md_lines)))
        else:
            cells.append(nbf.v4.new_code_cell("".join(current)))
        current = []

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("# %%"):
            flush()
            cell_type = "markdown" if "[markdown]" in stripped.lower() else "code"
            current = []
            continue
        current.append(line)

    flush()
    return cells


def convert_file(py_path: Path) -> Path:
    target = py_path.with_suffix(".ipynb")
    lines = py_path.read_text().splitlines(keepends=True)
    cells = parse_cells(lines)

    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
    )

    target.write_text(nbf.writes(nb))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert tutorial .py files to notebooks.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("notebooks/tutorials"),
        help="Directory to scan for tutorial .py files.",
    )
    args = parser.parse_args()

    py_files = sorted(args.root.glob("*.py"))
    if not py_files:
        raise SystemExit(f"No .py tutorials found under {args.root}")

    for py_file in py_files:
        target = convert_file(py_file)
        print(f"Converted {py_file} -> {target}")


if __name__ == "__main__":
    main()
