"""Fix a flattened project download and report what is in the dataset.

If you downloaded the project files one at a time, they all landed in a single
folder with no `src/` or `configs/` subdirectory. Python then cannot import
`src.smoke_test`, because there is no package called `src`.

This script repairs the layout in place. It is safe to run more than once --
anything already in the right place is left alone.

Pure standard library. Run it BEFORE `pip install -r requirements.txt`.

    python setup_project.py
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

SRC_MODULES = [
    "analyze.py", "build_index.py", "datasets.py", "engine.py",
    "extract_frames.py", "gradcam.py", "make_budgets.py", "metrics.py",
    "models.py", "run_experiment.py", "sampling.py", "smoke_test.py",
    "smoothing.py", "splits.py", "utils.py",
]
CONFIG_FILES = ["default.yaml", "gru.yaml", "causal.yaml"]
ROOT_FILES = ["README.md", "PROTOCOL.md", "RUNBOOK.md", "requirements.txt",
              "run_all.sh"]

root = Path(__file__).resolve().parent
moved, made, problems = [], [], []


def step(msg):
    print(f"\n{msg}")


# ---------------------------------------------------------------------------
step("1. creating package folders")
for d in ("src", "configs", "data"):
    p = root / d
    if not p.exists():
        p.mkdir(parents=True)
        made.append(f"{d}/")
        print(f"   created {d}/")
    else:
        print(f"   {d}/ already exists")

# ---------------------------------------------------------------------------
step("2. moving python modules into src/")
for name in SRC_MODULES:
    loose, target = root / name, root / "src" / name
    if target.exists() and not loose.exists():
        continue
    if loose.exists():
        if target.exists():
            target.unlink()
        shutil.move(str(loose), str(target))
        moved.append(f"{name} -> src/")
        print(f"   moved {name}")
    else:
        problems.append(f"MISSING: {name} (not in root, not in src/)")

# ---------------------------------------------------------------------------
step("3. creating src/__init__.py")
init = root / "src" / "__init__.py"
if not init.exists():
    # This empty file is what makes `src` an importable package. Without it,
    # `python -m src.smoke_test` fails with ModuleNotFoundError.
    init.write_text("")
    made.append("src/__init__.py")
    print("   created src/__init__.py")
else:
    print("   already present")

# ---------------------------------------------------------------------------
step("4. moving configs into configs/")
for name in CONFIG_FILES:
    loose, target = root / name, root / "configs" / name
    if loose.exists() and not target.exists():
        shutil.move(str(loose), str(target))
        moved.append(f"{name} -> configs/")
        print(f"   moved {name}")

default_cfg = root / "configs" / "default.yaml"
if not default_cfg.exists():
    problems.append("MISSING: configs/default.yaml -- nothing will run without it")
    print("   !! configs/default.yaml not found")

# ---------------------------------------------------------------------------
step("5. generating the two ablation configs")
if default_cfg.exists():
    base = default_cfg.read_text()

    variants = {
        # GRU instead of LSTM, everything else identical
        "gru.yaml": [(r"^(\s*rnn:\s*)\S+(.*)$", r"\1gru\2")],
        # unidirectional: cannot see future frames, so it is the only variant
        # deployable during a live scan
        "causal.yaml": [(r"^(\s*bidirectional:\s*)\S+(.*)$", r"\1false\2")],
    }

    for fname, subs in variants.items():
        target = root / "configs" / fname
        if target.exists():
            print(f"   {fname} already exists, leaving it")
            continue
        text = base
        for pat, rep in subs:
            text, n = re.subn(pat, rep, text, count=1, flags=re.MULTILINE)
            if n == 0:
                problems.append(f"could not patch {fname} (pattern {pat!r} not found)")
        target.write_text(text)
        made.append(f"configs/{fname}")
        print(f"   created configs/{fname}")

# ---------------------------------------------------------------------------
step("6. locating the dataset")
candidates = [root / "data" / "DatasetV3", root / "DatasetV3"]
dataset = next((c for c in candidates if c.exists() and c.is_dir()), None)

if dataset is None:
    print("   no DatasetV3/ folder found yet -- that is fine, you do not need")
    print("   it for the offline smoke test.")
elif dataset == root / "DatasetV3":
    target = root / "data" / "DatasetV3"
    if target.exists():
        print(f"   both {dataset} and {target} exist -- leaving both alone,")
        print("   check manually which one has the real content")
    else:
        print("   moving DatasetV3/ into data/ (this may take a minute)")
        shutil.move(str(dataset), str(target))
        dataset = target
        moved.append("DatasetV3/ -> data/")
        print("   done")
else:
    print("   already at data/DatasetV3")

# ---------------------------------------------------------------------------
step("7. dataset inventory")
if dataset and dataset.exists():
    # Walk it and summarise by extension and top-level folder. This is the
    # information needed to confirm whether the frames are already extracted
    # and where the label CSV lives.
    by_ext: dict[str, int] = {}
    by_dir: dict[str, int] = {}
    csvs: list[Path] = []
    total = 0

    for p in dataset.rglob("*"):
        if not p.is_file():
            continue
        total += 1
        ext = p.suffix.lower() or "(none)"
        by_ext[ext] = by_ext.get(ext, 0) + 1
        rel = p.relative_to(dataset)
        key = "/".join(rel.parts[:2]) if len(rel.parts) > 1 else "(root)"
        by_dir[key] = by_dir.get(key, 0) + 1
        if ext == ".csv":
            csvs.append(p)

    print(f"   {total} files under {dataset}")
    print("\n   by extension:")
    for ext, n in sorted(by_ext.items(), key=lambda kv: -kv[1])[:12]:
        print(f"     {ext:<12} {n:>8}")
    print("\n   by folder:")
    for d, n in sorted(by_dir.items(), key=lambda kv: -kv[1])[:20]:
        print(f"     {d:<40} {n:>8}")

    print("\n   sample filenames:")
    for p in list(dataset.rglob("*"))[:8]:
        if p.is_file():
            print(f"     {p.relative_to(dataset)}")

    if csvs:
        print("\n   CSV files found (these hold the labels):")
        for c in csvs[:10]:
            print(f"     {c.relative_to(dataset)}")
            try:
                with open(c, encoding="utf-8", errors="replace") as f:
                    header = f.readline().strip()
                    rows = [f.readline().strip() for _ in range(3)]
                print(f"       header: {header}")
                for r in rows:
                    if r:
                        print(f"       row:    {r}")
            except Exception as e:
                print(f"       (could not read: {e})")
    else:
        print("\n   !! no CSV found -- labels may be in JSON or TXT. "
              "Check the extension list above.")

# ---------------------------------------------------------------------------
step("8. verification")
ok = True
for name in SRC_MODULES + ["__init__.py"]:
    if not (root / "src" / name).exists():
        print(f"   [FAIL] src/{name} missing")
        ok = False
if (root / "configs" / "default.yaml").exists():
    print("   [PASS] configs/default.yaml")
else:
    print("   [FAIL] configs/default.yaml missing")
    ok = False
if ok:
    print(f"   [PASS] all {len(SRC_MODULES) + 1} files present in src/")

print("\n" + "=" * 62)
if moved:
    print(f"moved {len(moved)} item(s), created {len(made)} item(s)")
if problems:
    print("\nPROBLEMS:")
    for p in problems:
        print("  -", p)
    print("\nRe-download the missing files, put them in this folder, "
          "and run this script again.")
    sys.exit(1)

print("\nLayout is correct. Next:")
print("    pip install numpy pandas scikit-learn Pillow PyYAML "
      "opencv-python-headless matplotlib")
print("    python -m src.smoke_test --synthetic")
