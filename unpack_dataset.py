"""Unwrap DatasetV3's nested archives and report what is really inside.

DatasetV3.zip is a Google Drive bulk export: a zip whose contents are three
more zips (train / val / test) plus README files. The actual videos and labels
are one layer deeper, so the ordinary extract leaves you with archives, not
data.

This script:
  1. finds the extracted outer folder, wherever it ended up
  2. prints the dataset's own README_EN.md -- that is the authoritative
     description of the label format, and worth reading before anything else
  3. inspects each inner zip WITHOUT extracting, including reading any label
     files straight out of the archive
  4. with --extract, unpacks all three into a clean data/DatasetV3/ layout
     and flattens the doubled DatasetV3/DatasetV3/ nesting

Usage:
    python unpack_dataset.py
    python unpack_dataset.py --extract
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

TEXT_EXTS = {".csv", ".json", ".txt"}


def long_path(p: Path) -> str:
    """Windows extended-length path -- bypasses the 260-char MAX_PATH limit."""
    p = p.resolve()
    if os.name != "nt":
        return str(p)
    s = str(p)
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):
        return "\\\\?\\UNC\\" + s.lstrip("\\")
    return "\\\\?\\" + s


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def find_root(start: Path) -> Path | None:
    """Locate the folder holding the inner zips, however deeply it nested."""
    for cand in (start, start / "DatasetV3", start / "DatasetV3" / "DatasetV3"):
        if cand.is_dir() and any(cand.glob("*.zip")):
            return cand
    for p in start.rglob("*.zip"):
        return p.parent
    return None


def show_readme(root: Path) -> None:
    """The dataset's own documentation. Read this before writing any code."""
    for name in ("README_EN.md", "README.md", "README_CN.md"):
        p = root / name
        if not p.exists():
            continue
        print("\n" + "=" * 66)
        print(f"  {name}  -- the dataset's own documentation")
        print("=" * 66)
        try:
            print(p.read_text(encoding="utf-8", errors="replace")[:6000])
        except Exception as e:
            print(f"(could not read: {e})")
        return
    print("\n(no README found next to the inner zips)")


def inspect_inner(zp: Path) -> list[zipfile.ZipInfo]:
    print("\n" + "-" * 66)
    print(f"  {zp.name}   ({human(zp.stat().st_size)})")
    print("-" * 66)
    try:
        zf = zipfile.ZipFile(zp)
    except zipfile.BadZipFile as e:
        print(f"  CORRUPT: {e}")
        return []

    infos = [i for i in zf.infolist() if not i.is_dir()]
    print(f"  {len(infos)} files, {human(sum(i.file_size for i in infos))} uncompressed")

    by_ext: dict[str, list[int]] = {}
    by_dir: dict[str, int] = {}
    for i in infos:
        ext = Path(i.filename).suffix.lower() or "(none)"
        by_ext.setdefault(ext, []).append(i.file_size)
        parts = i.filename.replace("\\", "/").split("/")
        key = "/".join(parts[:-1]) if len(parts) > 1 else "(root)"
        by_dir[key] = by_dir.get(key, 0) + 1

    print("\n  by extension:")
    for ext, sizes in sorted(by_ext.items(), key=lambda kv: -len(kv[1])):
        print(f"    {ext:<10} {len(sizes):>7} files  {human(sum(sizes)):>9}")

    print("\n  folders:")
    for d, n in sorted(by_dir.items(), key=lambda kv: -kv[1])[:15]:
        print(f"    {d or '(root)':<50} {n:>7}")

    print("\n  sample filenames:")
    for i in infos[:8]:
        print(f"    {i.filename}")

    # read label files directly out of the archive -- no extraction needed
    labels = [i for i in infos
              if Path(i.filename).suffix.lower() in TEXT_EXTS
              and i.file_size < 50_000_000]
    if labels:
        print(f"\n  {len(labels)} label/text file(s):")
        for i in labels[:8]:
            print(f"\n    --- {i.filename}  ({human(i.file_size)}) ---")
            try:
                with zf.open(i) as fh:
                    head = fh.read(3000).decode("utf-8", errors="replace")
                for line in head.splitlines()[:8]:
                    print(f"        {line[:150]}")
            except Exception as e:
                print(f"        (unreadable: {e})")
    else:
        print("\n  no CSV/JSON/TXT inside this archive")

    zf.close()
    return infos


def extract_inner(zp: Path, dest: Path) -> int:
    zf = zipfile.ZipFile(zp)
    infos = [i for i in zf.infolist() if not i.is_dir()]
    dest.mkdir(parents=True, exist_ok=True)
    target = long_path(dest)
    print(f"\n  extracting {zp.name} -> {dest}  ({len(infos)} files)")

    failures = []
    for n, info in enumerate(infos, 1):
        try:
            zf.extract(info, target)
        except Exception as e:
            failures.append((info.filename, repr(e)))
        if n % 100 == 0 or n == len(infos):
            print(f"\r    {n}/{len(infos)} ({100 * n / len(infos):.0f}%)",
                  end="", flush=True)
    print()
    zf.close()

    if failures:
        print(f"    {len(failures)} FAILED:")
        for f, e in failures[:5]:
            print(f"      {f}\n        {e}")
    return len(failures)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/DatasetV3",
                    help="Where the outer zip was extracted")
    ap.add_argument("--extract", action="store_true")
    a = ap.parse_args()

    start = Path(a.dir)
    if not start.exists():
        print(f"{start} does not exist. Extract the outer zip first:")
        print("    python inspect_zip.py --extract data/DatasetV3")
        return 1

    root = find_root(start)
    if root is None:
        print(f"No inner .zip files found under {start}.")
        print("Contents:")
        for p in list(start.rglob("*"))[:20]:
            print("   ", p.relative_to(start))
        return 1

    print(f"inner archives found in: {root.resolve()}")
    inner = sorted(root.glob("*.zip"))
    for z in inner:
        print(f"  {z.name}  ({human(z.stat().st_size)})")

    show_readme(root)

    for z in inner:
        inspect_inner(z)

    if not a.extract:
        print("\n" + "=" * 66)
        print("Inspection only. To unpack all three:")
        print(f"    python unpack_dataset.py --dir {a.dir} --extract")
        return 0

    # ---- extract ---------------------------------------------------------
    # Everything lands directly in data/DatasetV3/, so the inner zips'
    # own train/ val/ test/ prefixes produce the layout the rest of the
    # pipeline expects.
    final = Path("data/DatasetV3")
    final.mkdir(parents=True, exist_ok=True)
    total_fail = sum(extract_inner(z, final) for z in inner)

    # copy the documentation across too
    for name in ("README.md", "README_EN.md", "README_CN.md"):
        src = root / name
        if src.exists() and not (final / name).exists():
            shutil.copy2(src, final / name)

    print("\n" + "=" * 66)
    n_files = sum(1 for p in final.rglob("*") if p.is_file())
    print(f"{n_files} files now under {final.resolve()}")

    by_ext: dict[str, int] = {}
    for p in final.rglob("*"):
        if p.is_file():
            by_ext[p.suffix.lower() or "(none)"] = \
                by_ext.get(p.suffix.lower() or "(none)", 0) + 1
    print("\nby extension:")
    for ext, n in sorted(by_ext.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {ext:<10} {n:>8}")

    print("\ntop-level folders:")
    for p in sorted(final.iterdir()):
        if p.is_dir():
            print(f"  {p.name}/  ({sum(1 for _ in p.rglob('*') if _.is_file())} files)")

    if total_fail:
        print(f"\n{total_fail} file(s) failed to extract -- see above.")
        return 1

    # The inner zips are still on disk alongside the extracted data, which
    # means ~1.1 GB is duplicated. Point that out rather than deleting it
    # automatically -- if an extract went wrong, those archives are the only
    # copy left.
    leftover = root if root.resolve() != final.resolve() else None
    if leftover and leftover.exists():
        size = sum(p.stat().st_size for p in leftover.rglob("*") if p.is_file())
        print(f"\nThe original inner archives are still at:\n    {leftover.resolve()}")
        print(f"    ({human(size)} duplicated)")
        print("Once you have confirmed the counts above look right, you can "
              "delete that folder to reclaim the space.")

    print("\nAll archives unpacked. Next:")
    print("    python -m src.extract_frames --dataset-root data/DatasetV3 "
          "--out-dir data/frames")
    print("    python -m src.build_index --dataset-root data/DatasetV3 "
          "--frames-dir data/frames --out data/index.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
