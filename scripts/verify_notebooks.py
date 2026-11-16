"""
Validation and verification suite for the 6 Jupyter Notebooks.
"""

import json
import os
import nbformat

notebooks = [
    "01_dataset_analysis.ipynb",
    "02_preprocessing.ipynb",
    "03_logistic_regression.ipynb",
    "04_svm_model.ipynb",
    "05_distilbert_training.ipynb",
    "06_model_comparison.ipynb"
]

all_valid = True

for nb_name in notebooks:
    path = os.path.join("notebooks", nb_name)
    print(f"\\nValidating {nb_name}...")
    
    # 1. Check JSON validity
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("  [PASS] Valid JSON syntax.")
    except Exception as e:
        print(f"  [FAIL] JSON parse error: {e}")
        all_valid = False
        continue

    # 2. Check nbformat schema validation
    try:
        nb_obj = nbformat.from_dict(data)
        nbformat.validate(nb_obj)
        print(f"  [PASS] nbformat schema validation passed (format {data['nbformat']}.{data['nbformat_minor']}).")
    except Exception as e:
        print(f"  [FAIL] nbformat schema validation error: {e}")
        all_valid = False

    # 3. Check metadata
    meta = data.get("metadata", {})
    ks = meta.get("kernelspec", {})
    li = meta.get("language_info", {})
    if ks.get("display_name") == "Python 3" and ks.get("language") == "python" and ks.get("name") == "python3":
        print("  [PASS] kernelspec matches specification.")
    else:
        print(f"  [FAIL] kernelspec mismatch: {ks}")
        all_valid = False

    if li.get("name") == "python" and li.get("version") == "3.10.0":
        print("  [PASS] language_info matches specification.")
    else:
        print(f"  [FAIL] language_info mismatch: {li}")
        all_valid = False

    if data.get("nbformat") == 4 and data.get("nbformat_minor") == 5:
        print("  [PASS] nbformat version is exactly 4.5.")
    else:
        print(f"  [FAIL] nbformat version mismatch: {data.get('nbformat')}.{data.get('nbformat_minor')}")
        all_valid = False

    # 4. Cell diagnostics
    cells = data.get("cells", [])
    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    print(f"  [INFO] Total cells: {len(cells)} ({len(md_cells)} markdown, {len(code_cells)} code)")

    # Validate each cell structure
    cell_err = False
    for i, c in enumerate(cells):
        ctype = c.get("cell_type")
        if ctype not in ["markdown", "code"]:
            print(f"  [FAIL] Cell {i} invalid cell_type: {ctype}")
            cell_err = True
        if "source" not in c or not isinstance(c["source"], list):
            print(f"  [FAIL] Cell {i} missing valid 'source' list")
            cell_err = True
        if ctype == "code":
            if "execution_count" not in c or "outputs" not in c or not isinstance(c["outputs"], list):
                print(f"  [FAIL] Code cell {i} missing execution_count or outputs list")
                cell_err = True
    if not cell_err:
        print("  [PASS] All cell data structures valid.")
    else:
        all_valid = False

print(f"\\nOverall Verification Result: {'SUCCESS - ALL 6 NOTEBOOKS VALID' if all_valid else 'FAILURE'}")
