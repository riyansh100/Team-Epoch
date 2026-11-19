"""Execute a notebook headlessly (used by tests/test_pipeline.py).

    python tests/run_notebook.py notebooks/01_build_ntcir_ddg.ipynb executed.ipynb

Environment variables (NTCIR_DDG_FIXTURE=1, NTCIR_DDG_WORK_DIR, ...) are passed through to the kernel.
Prints every cell's text output; exits non-zero on the first failing cell.
"""

from __future__ import annotations

import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def main(src: str, dst: str) -> int:
    nb = nbformat.read(src, as_version=4)
    client = NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=False)
    rc = 0
    try:
        client.execute()
    except CellExecutionError as e:
        print(str(e)[-4000:])
        rc = 1
    finally:
        nbformat.write(nb, dst)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            text = out.get("text") or (out.get("data", {}) or {}).get("text/plain", "")
            if text:
                print(f"--- cell {i} ---\n{text if isinstance(text, str) else ''.join(text)}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
