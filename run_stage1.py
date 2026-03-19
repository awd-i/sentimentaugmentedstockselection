"""Run Stage 1 of the pipeline (01_data_prep.ipynb logic) as a standalone script."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks" / "01_data_prep.ipynb"

def main():
    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)
    code_cells = [
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code"
    ]
    import os
    os.chdir(ROOT)
    globals_dict = {}
    for i, code in enumerate(code_cells):
        print(f"--- Running cell {i} ---")
        try:
            exec(compile(code, f"<cell {i}>", "exec"), globals_dict)
        except Exception as e:
            print(f"Cell {i} failed: {e}")
            raise
    print("Stage 1 complete.")

if __name__ == "__main__":
    main()
