"""Inspect DatasetV3.zip without extracting it, and extract it reliably.

Two problems this solves.

1. Windows `Expand-Archive` fails silently when an extracted path would exceed
   260 characters. It creates the destination folder, writes nothing, and
   returns with no error. If your DatasetV3/ folder is empty, this is almost
   certainly why. Python's zipfile can be pointed at an extended-length path
   (`\\\\?\\C:\\...`) which lifts that limit.

2. You do not need to extract anything to find out what is inside. A zip's
   central directory lists every filename, and small files like the label CSV
   can be read straight out of the archive. That is enough to confirm the
   layout and the label column names before committing to a 1 GB extract.

Usage:
    python inspect_zip.py                          # inspect only
    python inspect_zip.py --extract data/DatasetV3 # inspect, then extract
    python inspect_zip.py --zip path/to/other.zip
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path


def long_path(p: Path) -> str:
    """Windows extended-length path, which bypasses the 260-char MAX_PATH."""
    p = p.resolve()
    if os.name != "nt":
        return str(p)
    s = str(p)
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):                     # UNC share
        return "\\\\?\\UNC\\" + s.lstrip("\\")
    return "\\\\?\\" + s


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def inspect(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos = [i for i in zf.infolist() if not i.is_dir()]
    total = sum(i.file_size for i in infos)
    print(f"\n{len(infos)} files, {human(total)} uncompressed\n")

    by_ext: dict[str, list[int]] = {}
    by_dir: dict[str, int] = {}
    for i in infos:
        ext = Path(i.filename).suffix.lower() or "(none)"
        by_ext.setdefault(ext, []).append(i.file_size)
        parts = i.filename.replace("\\", "/").split("/")
        key = "/".join(parts[:2]) if len(parts) > 2 else (
            parts[0] if len(parts) > 1 else "(root)")
        by_dir[key] = by_dir.get(key, 0) + 1

    print("by extension:")
    for ext, sizes in sorted(by_ext.items(), key=lambda kv: -len(kv[1])):
        print(f"  {ext:<10} {len(sizes):>8} files   {human(sum(sizes)):>9}")

    print("\nby folder:")
    for d, n in sorted(by_dir.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {d:<45} {n:>8}")

    print("\nfirst 15 filenames:")
    for i in infos[:15]:
        print(f"  {i.filename}")

    # ---- the important part: read the label files straight out of the zip --
    label_like = [i for i in infos
                  if Path(i.filename).suffix.lower() in (".csv", ".json", ".txt")
                  and i.file_size < 50_000_000]
    if label_like:
        print(f"\n{len(label_like)} candidate label file(s):")
        for i in label_like[:12]:
            print(f"\n  --- {i.filename}  ({human(i.file_size)}) ---")
            try:
                with zf.open(i) as fh:
                    head = fh.read(4000).decode("utf-8", errors="replace")
                for line in head.splitlines()[:6]:
                    print(f"      {line[:160]}")
            except Exception as e:
                print(f"      (could not read: {e})")
    else:
        print("\n!! no CSV/JSON/TXT found -- labels may be embedded elsewhere.")

    # ---- warn about the exact thing that broke Expand-Archive -------------
    longest = max(infos, key=lambda i: len(i.filename))
    print(f"\nlongest path inside the zip: {len(longest.filename)} chars")
    print(f"  {longest.filename}")
    return infos


def extract(zf: zipfile.ZipFile, infos: list[zipfile.ZipInfo], dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    base_len = len(str(dest.resolve()))
    worst = base_len + max(len(i.filename) for i in infos) + 1
    print(f"\nextracting {len(infos)} files to {dest}")
    print(f"longest resulting path: ~{worst} chars", end="")
    if worst > 260 and os.name == "nt":
        print("  (over the 260 limit -- using extended-length paths)")
    else:
        print()

    target = long_path(dest)
    failures = []
    for n, info in enumerate(infos, 1):
        try:
            zf.extract(info, target)
        except Exception as e:
            failures.append((info.filename, repr(e)))
        if n % 250 == 0 or n == len(infos):
            pct = 100 * n / len(infos)
            print(f"\r  {n}/{len(infos)}  ({pct:.0f}%)", end="", flush=True)
    print()

    if failures:
        print(f"\n{len(failures)} file(s) failed:")
        for f, e in failures[:10]:
            print(f"  {f}\n    {e}")
    else:
        print("all files extracted cleanly")

    on_disk = sum(1 for p in dest.rglob("*") if p.is_file())
    print(f"\nverification: {on_disk} files now on disk under {dest}")
    if on_disk != len(infos):
        print(f"  !! expected {len(infos)} -- something did not land")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="DatasetV3.zip")
    ap.add_argument("--extract", metavar="DEST", default=None,
                    help="Extract to this folder after inspecting")
    a = ap.parse_args()

    zp = Path(a.zip)
    if not zp.exists():
        # look in the usual places before giving up
        for alt in (Path("data") / zp.name, Path("..") / zp.name):
            if alt.exists():
                zp = alt
                break
        else:
            print(f"Could not find {a.zip}. Pass --zip with the full path.")
            return 1

    print(f"reading {zp.resolve()}  ({human(zp.stat().st_size)})")
    try:
        zf = zipfile.ZipFile(zp)
    except zipfile.BadZipFile as e:
        print(f"\nThe zip is corrupt: {e}")
        print("Re-download it from https://zenodo.org/records/17655183")
        return 1

    bad = zf.testzip()
    if bad is not None:
        print(f"\nCorrupt entry inside the archive: {bad}")
        print("Re-download it.")
        return 1

    infos = inspect(zf)

    if a.extract:
        return extract(zf, infos, Path(a.extract))

    print("\n" + "=" * 62)
    print("Inspection only -- nothing extracted. To extract:")
    print(f"    python inspect_zip.py --zip {zp} --extract data/DatasetV3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
