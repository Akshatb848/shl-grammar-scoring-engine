"""Lightweight repository checks (no GPU, no competition data, no PyTorch).

- the committed notebook is exactly what build_notebook.py generates (cell for cell);
- every notebook code cell and every script is valid Python;
- submission.csv has the format Kaggle expects and scores in the 0-5 range.
"""

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "shl_grammar_scoring.ipynb"


def _cells(path: Path) -> list[tuple[str, str]]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for cell in nb["cells"]:
        src = cell["source"]
        out.append((cell["cell_type"], "".join(src) if isinstance(src, list) else src))
    return out


def test_notebook_matches_its_builder(tmp_path):
    subprocess.run(  # noqa: S603 - fixed interpreter and script path
        [sys.executable, str(ROOT / "build_notebook.py")], cwd=tmp_path, check=True
    )
    assert _cells(tmp_path / NOTEBOOK.name) == _cells(NOTEBOOK)


def test_notebook_code_cells_parse():
    for i, (kind, src) in enumerate(_cells(NOTEBOOK)):
        if kind == "code":
            # IPython magics / shell escapes (column 0) are not Python syntax.
            lines = [
                "pass" if line.startswith(("%", "!")) else line for line in src.splitlines()
            ]
            ast.parse("\n".join(lines), filename=f"cell {i}")


def test_scripts_parse():
    for name in ("build_notebook.py", "transcribe.py", "finetune_deberta.py"):
        ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)


def test_submission_format():
    with (ROOT / "submission.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows and list(rows[0]) == ["filename", "label"]
    names = [r["filename"] for r in rows]
    assert len(names) == len(set(names)) == 216
    assert all(n.endswith(".wav") for n in names)
    assert all(0.0 <= float(r["label"]) <= 5.0 for r in rows)
