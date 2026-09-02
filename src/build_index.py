"""Build ONE frame-level index that every experiment reads from.

Everything downstream (splits, sampling, datasets) consumes this single CSV:

    frame_path,video_id,frame_idx,label,orig_split

  frame_path : path to the extracted jpg, relative to --frames-dir
  video_id   : the video this frame came from  -> the grouping key for splits
  frame_idx  : 0-based position within the video -> the ordering key for clips
  label      : 1 = standard plane, 0 = non-standard
  orig_split : which official folder the video came from (train/validation/test)

Two things this file is deliberately strict about:

1. Every invariant is an assertion, not a log line. It refuses to write the
   index if any split's label join rate is below 1.0, if an extracted folder is
   empty or disagrees with the manifest, if an ALL/NONE sentinel fails to
   partition its video, or if two label files disagree about the same frame.
   Each of those failed silently at some point and produced plausible numbers
   rather than an error. Use --allow-unlabelled / --allow-empty-dirs to
   override, but only once you know why the shortfall is genuine.

2. Column names in class_label.csv are auto-detected and then PRINTED, along
   with the annotation granularity of each split. Check the printout against
   the real file before trusting anything downstream -- a silent mis-parse here
   poisons every number in the study.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .utils import get_logger

log = get_logger("index")

# Candidate column names, most specific first.
VIDEO_COLS = ["video_id", "video", "video_name", "case", "case_id", "vid"]
FRAME_COLS = ["frame_idx", "frame_id", "frame", "frame_number", "index", "idx"]
LABEL_COLS = ["label", "class", "cls", "standard_plane", "sp", "y", "target"]
PATH_COLS = ["image", "image_name", "filename", "file_name", "path", "img", "name"]

# Label strings that mean "standard plane".
POSITIVE_TOKENS = {"1", "standard", "standard_plane", "sp", "true", "yes", "std"}
NEGATIVE_TOKENS = {"0", "non-standard", "nonstandard", "non_standard", "ns",
                   "false", "no", "background", "other"}


def _pick(cols: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    # fall back to substring match
    for cand in candidates:
        for lc, orig in lower.items():
            if cand in lc:
                return orig
    return None


def normalise_label(v) -> int:
    """Map whatever the CSV says onto {0, 1}. Raise loudly on anything else."""
    s = str(v).strip().lower()
    if s in POSITIVE_TOKENS:
        return 1
    if s in NEGATIVE_TOKENS:
        return 0
    # Numeric multi-class labels: the IUGC convention has 0 = non-standard and
    # nonzero = a standard-plane subtype. Collapse to binary.
    try:
        f = float(s)
        return 1 if f > 0 else 0
    except ValueError:
        pass
    raise ValueError(
        f"Cannot interpret label {v!r}. Add it to POSITIVE_TOKENS/"
        f"NEGATIVE_TOKENS in build_index.py after checking the dataset README."
    )


def _parse_frame_ref(s: str) -> tuple[str, int | None]:
    """Split something like 'video012_000345.jpg' or 'video012/000345.png'
    into (video_id, frame_idx). Returns (stem, None) if no index is present."""
    s = str(s).strip()
    p = Path(s)
    if len(p.parts) >= 2:
        return p.parts[-2], _int_or_none(p.stem)
    stem = p.stem
    m = re.match(r"^(.*?)[_\-](\d+)$", stem)
    if m:
        return m.group(1), int(m.group(2))
    return stem, None


def _int_or_none(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def parse_index_list(v, frame_count: int | None = None,
                     clip: bool = True) -> set[int]:
    """Turn one cell of `pos_index` / `neg_index` into a set of frame indices.

    The dataset stores labels per VIDEO, listing which frame numbers are
    standard planes rather than giving one row per frame. How those lists are
    written varies between releases, so every plausible spelling is handled
    and the caller reports which one was seen:

        3                    single index
        1,2,3   /  1 2 3     separated
        [1, 2, 3]            bracketed list
        10-20   /  10:20     inclusive range
        [[1,5],[10,15]]      list of ranges
        ALL                  every frame of the video   (needs frame_count)
        NONE                 no frames of this class
        (empty / nan)        no frames of this class

    A range is read as INCLUSIVE of both endpoints. If that is wrong for this
    dataset it costs one frame at each boundary -- check the reported counts
    against `frame_count` before trusting it.

    `clip` filters the result to [0, frame_count). Pass clip=False when the
    caller wants to count out-of-range indices itself as a 1-based-CSV probe;
    the ALL/NONE sentinels are still resolved either way.

    ALL is not optional to support. train/cls/class_label.csv uses it for every
    standard-plane video in the split -- 266 of 434 -- and the string is not a
    Python literal, so without this branch it falls through to the digit scan,
    matches nothing, and returns an empty set. That is silent: the video simply
    contributes no labelled frames, the join rate drops, and training proceeds
    on whatever is left.
    """
    import ast
    import math
    import re

    if v is None:
        return set()
    if isinstance(v, float) and math.isnan(v):
        return set()

    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null", "[]", "-1"):
        return set()

    if s.upper() == "ALL":
        if frame_count is None:
            raise ValueError(
                "the 'ALL' sentinel means 'every frame of this video' and "
                "cannot be expanded without frame_count. Pass the video's "
                "frame count from the CSV's frame_count column.")
        return set(range(int(frame_count)))

    out: set[int] = set()

    # 1. Try to read it as a Python/JSON literal -- covers [1,2,3] and
    #    [[1,5],[10,15]] exactly, without regex guesswork.
    try:
        lit = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        lit = None

    if lit is not None:
        def walk(x):
            if isinstance(x, (int, float)):
                out.add(int(x))
            elif isinstance(x, (list, tuple)):
                # a 2-element list of ints is ambiguous: [10, 15] could be two
                # indices or the range 10..15. Nested inside another list it is
                # conventionally a range, so treat it as one only at depth >= 2.
                if len(x) == 2 and all(isinstance(i, (int, float)) for i in x) \
                        and isinstance(lit, (list, tuple)) \
                        and any(isinstance(e, (list, tuple)) for e in lit):
                    a, b = int(x[0]), int(x[1])
                    out.update(range(min(a, b), max(a, b) + 1))
                else:
                    for i in x:
                        walk(i)
        walk(lit)
        if out:
            return out

    # 2. Fall back to scanning for ranges and bare integers.
    for m in re.finditer(r"(\d+)\s*[-:~]\s*(\d+)", s):
        a, b = int(m.group(1)), int(m.group(2))
        out.update(range(min(a, b), max(a, b) + 1))
    consumed = re.sub(r"\d+\s*[-:~]\s*\d+", " ", s)
    out.update(int(m) for m in re.findall(r"\d+", consumed))

    if clip and frame_count is not None:
        out = {i for i in out if 0 <= i < frame_count}
    return out


def canonical_video_id(stem: str) -> str:
    """Collapse an exactly-doubled stem: 'X__X' -> 'X'.

    The DatasetV3 deposit ships 266 of its 434 train videos under a doubled
    filename -- train/videos/20190909T155747I1__20190909T155747I1.avi -- while
    train/cls/class_label.csv refers to the same video as
    20190909T155747I1.avi. The doubling is inside the published zip, so it is
    not something the unpack or extract step can be fixed to avoid; it has to
    be reconciled on the join.

    The match must be exact halves. Legitimate filenames in this dataset
    contain '__' as an ordinary separator (20190830T115644__B_产科_tmp_0), and
    splitting on the first occurrence would mangle those. Requiring
    s == h + '__' + h leaves them untouched.
    """
    s = str(stem).strip()
    sep = "__"
    n = len(s) - len(sep)
    if n > 0 and n % 2 == 0:
        h = n // 2
        if s[h:h + len(sep)] == sep and s[:h] == s[h + len(sep):]:
            return s[:h]
    return s


# Column-name variants for the per-video frame-index format.
#
# These must be SPECIFIC. An earlier version accepted the bare names "pos" and
# "sp", which matched a column in train_info.csv that holds something else
# entirely (one scalar per video, not a frame list) and manufactured one fake
# positive per video. A label parser that guesses wrong does not crash -- it
# produces a plausible-looking table and quietly corrupts every downstream
# number. So: only names that unambiguously mean "list of frame indices".
POS_INDEX_COLS = ["sp_index", "pos_index", "standard_index", "positive_index",
                  "sp_frames", "positive_frames"]
NEG_INDEX_COLS = ["nsp_index", "neg_index", "nonstandard_index",
                  "non_standard_index", "negative_index", "nsp_frames",
                  "negative_frames"]
FILENAME_COLS = ["filename", "file_name", "video", "video_name", "video_id",
                 "name", "case", "case_id"]
SEG_FRAME_COLS = ["labeled_frame_index", "labelled_frame_index",
                  "annotated_frame_index", "seg_frame_index"]

# Columns that look label-ish but are NOT per-frame class labels. Matching any
# of these in the per-frame path means the file is metadata, not labels.
NOT_A_LABEL_COL = {"labeled_frame_count", "labelled_frame_count",
                   "labeled_frame_index", "labelled_frame_index",
                   "frame_count", "sp_count", "nsp_count", "n_frames"}


def load_video_index_table(csv_path: Path) -> pd.DataFrame:
    """Expand a per-video row with index lists into one row per frame.

    Frames listed in neither the positive nor the negative column are left OUT
    rather than assumed negative. Treating unannotated frames as non-standard
    would invent labels the annotators never gave and inflate the negative
    class. In this dataset the two lists happen to partition every frame, so
    nothing is usually dropped -- but the rule has to hold for the file where
    that is not true.
    """
    df = read_csv_any_encoding(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    fn_col = pick(FILENAME_COLS)
    pos_col = pick(POS_INDEX_COLS)
    neg_col = pick(NEG_INDEX_COLS)
    cnt_col = pick(["frame_count", "n_frames", "num_frames", "total_frames"])
    seg_col = pick(SEG_FRAME_COLS)

    if pos_col is None:
        raise SystemExit(
            f"{csv_path}: no positive-index column. Columns are "
            f"{list(df.columns)}. Add the right name to POS_INDEX_COLS.")
    if fn_col is None:
        raise SystemExit(
            f"{csv_path}: no filename/video column. Columns are "
            f"{list(df.columns)}. Add the right name to FILENAME_COLS.")

    log.info("    per-video index format: file=%r pos=%r neg=%r count=%r seg=%r",
             fn_col, pos_col, neg_col, cnt_col, seg_col)

    rows = []
    stats = {"videos": 0, "pos": 0, "neg": 0, "unannotated": 0,
             "out_of_range": 0, "overlap": 0, "sentinel": 0, "no_count": 0}

    def _is_sentinel(x) -> bool:
        return str(x).strip().upper() in ("ALL", "NONE")

    for _, r in df.iterrows():
        vid = canonical_video_id(Path(str(r[fn_col]).strip()).stem)
        fc = None
        if cnt_col is not None:
            try:
                fc = int(r[cnt_col])
            except (TypeError, ValueError):
                fc = None

        # ALL/NONE describe the whole video, so they are meaningless without a
        # frame count. Refuse rather than resolve them to an empty set.
        uses_sentinel = _is_sentinel(r[pos_col]) or (
            neg_col is not None and _is_sentinel(r[neg_col]))
        if uses_sentinel:
            stats["sentinel"] += 1
            if fc is None:
                stats["no_count"] += 1
                raise SystemExit(
                    f"{csv_path.name}: video {vid!r} uses an ALL/NONE sentinel "
                    f"but no frame-count column was found (looked for "
                    f"frame_count / n_frames / num_frames / total_frames). "
                    f"Columns are {list(df.columns)}.")

        pos_raw = parse_index_list(r[pos_col], fc, clip=False)
        neg_raw = (parse_index_list(r[neg_col], fc, clip=False)
                   if neg_col else set())

        if fc is not None:
            stats["out_of_range"] += len(
                [i for i in (pos_raw | neg_raw) if not 0 <= i < fc])
            pos = {i for i in pos_raw if 0 <= i < fc}
            neg = {i for i in neg_raw if 0 <= i < fc}
            stats["unannotated"] += max(0, fc - len(pos | neg))
        else:
            pos, neg = pos_raw, neg_raw

        # A frame in both lists: positive wins. A frame an annotator marked
        # usable is usable.
        overlap_here = len(pos & neg)
        stats["overlap"] += overlap_here
        neg = neg - pos

        # ALL/NONE is a statement about every frame of the video, so the two
        # columns must be exact complements: disjoint, and covering all of it.
        #
        # Overlap is tolerated for ordinary frame lists (positive wins, above),
        # but not here. An explicit enumeration is more specific than a blanket
        # ALL, so a frame listed in both means the two columns contradict each
        # other -- and resolving that by letting the sentinel win would discard
        # the more specific annotation without saying so.
        if uses_sentinel:
            neg_repr = r[neg_col] if neg_col else None
            if overlap_here:
                raise SystemExit(
                    f"{csv_path.name}: video {vid!r} uses an ALL/NONE sentinel, "
                    f"but {overlap_here} frame(s) appear in BOTH columns. The "
                    f"sentinel and the explicit list disagree about the same "
                    f"frames. pos_index={r[pos_col]!r} neg_index={neg_repr!r}")
            if len(pos | neg) != fc:
                raise SystemExit(
                    f"{csv_path.name}: video {vid!r} uses an ALL/NONE sentinel "
                    f"but the resolved labels cover {len(pos | neg)} of {fc} "
                    f"frames. pos_index={r[pos_col]!r} neg_index={neg_repr!r}")

        seg_idx = None
        if seg_col is not None:
            try:
                seg_idx = int(r[seg_col])
            except (TypeError, ValueError):
                seg_idx = None

        for i in sorted(pos):
            rows.append((vid, i, 1, int(i == seg_idx)))
        for i in sorted(neg):
            rows.append((vid, i, 0, int(i == seg_idx)))

        stats["videos"] += 1
        stats["pos"] += len(pos)
        stats["neg"] += len(neg)

    log.info("    %d videos -> %d positive + %d negative labelled frames",
             stats["videos"], stats["pos"], stats["neg"])
    if stats["sentinel"]:
        log.info("    %d video(s) used an ALL/NONE sentinel (whole-video "
                 "label) and were expanded against frame_count",
                 stats["sentinel"])
    if stats["unannotated"]:
        log.info("    %d frames carry NO annotation and are excluded "
                 "(not assumed negative)", stats["unannotated"])
    if stats["overlap"]:
        log.warning("    %d frame(s) appeared in BOTH lists; treated as "
                    "positive", stats["overlap"])
    if stats["out_of_range"]:
        log.warning("    %d index value(s) fell outside [0, frame_count) and "
                    "were dropped -- the CSV may be 1-based.",
                    stats["out_of_range"])

    # A genuine frame-index list names many frames per video. Roughly one
    # index per row means the column held a scalar, not a list -- a metadata
    # field that happened to match a candidate name. Refuse it rather than
    # emit one fabricated label per video.
    n_labels = stats["pos"] + stats["neg"]
    if stats["videos"] and n_labels / stats["videos"] < 1.5:
        raise SystemExit(
            f"{csv_path.name}: only {n_labels} labels across {stats['videos']} "
            f"videos (~{n_labels / stats['videos']:.1f} per video). Column "
            f"{pos_col!r} is almost certainly a scalar metadata field, not a "
            f"frame-index list -- refusing to use it")

    return pd.DataFrame(rows, columns=["video_id", "frame_idx", "label",
                                       "is_seg_frame"])


def read_csv_any_encoding(csv_path: Path) -> pd.DataFrame:
    """Read a CSV whose encoding we do not know in advance.

    This dataset ships a README_CN.md, and its label CSV is not UTF-8 --
    pandas fails with `'utf-8' codec can't decode byte 0xb2`. Chinese-authored
    CSVs are usually GB18030 (the current national standard, a superset of
    GBK/GB2312). We try the plausible encodings in order of likelihood and
    report which one worked, so the choice is visible rather than silent.

    latin-1 is last and never fails -- every byte maps to some character. That
    makes it a safety net, not a correct answer: if it is what succeeds, the
    column names may contain mojibake, so check the printed header.
    """
    attempts = ["utf-8", "utf-8-sig", "gb18030", "big5", "cp1252", "latin-1"]
    last_err = None
    for enc in attempts:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            if enc != "utf-8":
                log.info("    (decoded as %s, not utf-8)", enc)
            return df
        except (UnicodeDecodeError, LookupError) as e:
            last_err = e
            continue
    raise SystemExit(f"Could not decode {csv_path} with any of {attempts}. "
                     f"Last error: {last_err}")


def load_label_table(csv_path: Path) -> pd.DataFrame:
    """Read one label CSV into (video_id, frame_idx, label).

    Two layouts exist in the wild and we detect which one this is:
      * per-VIDEO  -- one row per video with pos_index / neg_index frame lists
      * per-FRAME  -- one row per frame with a label column
    """
    df = read_csv_any_encoding(csv_path)
    log.info("  %s -> %d rows, columns: %s", csv_path.name, len(df),
             list(df.columns))

    lower = {c.lower().strip() for c in df.columns}
    if lower & set(POS_INDEX_COLS) or lower & set(NEG_INDEX_COLS):
        return load_video_index_table(csv_path)

    # Per-frame path. Require an EXACT label-column match, never a substring:
    # substring matching is how 'labeled_frame_count' (always 1) got mistaken
    # for a class label and produced a file of all-positive rows.
    exact = {c.lower().strip(): c for c in df.columns}
    label_col = next((exact[c] for c in LABEL_COLS
                      if c in exact and c not in NOT_A_LABEL_COL), None)
    if label_col is None:
        raise SystemExit(
            f"no per-frame label column in {csv_path.name} "
            f"(columns: {list(df.columns)}) -- treating it as metadata, "
            f"not labels")

    if df[label_col].nunique() < 2:
        raise SystemExit(
            f"{csv_path.name}: column {label_col!r} has only one distinct "
            f"value ({df[label_col].iloc[0]!r}) -- that is a metadata field, "
            f"not a class label")

    vid_col = _pick(list(df.columns), VIDEO_COLS)
    frm_col = _pick(list(df.columns), FRAME_COLS)
    path_col = _pick(list(df.columns), PATH_COLS)

    if vid_col and frm_col:
        out = pd.DataFrame({
            "video_id": df[vid_col].astype(str).map(
                lambda s: canonical_video_id(Path(str(s)).stem)),
            "frame_idx": df[frm_col].astype(int),
        })
    elif path_col:
        parsed = df[path_col].map(_parse_frame_ref)
        out = pd.DataFrame({
            "video_id": [canonical_video_id(a) for a, _ in parsed],
            "frame_idx": [b for _, b in parsed],
        })
        if out["frame_idx"].isna().any():
            raise SystemExit(
                f"Could not recover a frame index from {path_col} in {csv_path}. "
                f"Example value: {df[path_col].iloc[0]!r}. Extend _parse_frame_ref()."
            )
        out["frame_idx"] = out["frame_idx"].astype(int)
    else:
        raise SystemExit(
            f"{csv_path}: need either (video, frame) columns or a filename "
            f"column. Got {list(df.columns)}."
        )

    out["label"] = df[label_col].map(normalise_label).astype(int)
    log.info("    using video=%r frame=%r path=%r label=%r",
             vid_col, frm_col, path_col, label_col)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--frames-dir", required=True,
                    help="Output of extract_frames.py")
    ap.add_argument("--out", default="data/index.csv")
    ap.add_argument("--allow-empty-dirs", action="store_true",
                    help="Proceed even if some extracted video folders are "
                         "empty or disagree with manifest.csv. Off by default: "
                         "a missing video is a silent change to the study "
                         "population, not a warning.")
    ap.add_argument("--allow-unlabelled", action="store_true",
                    help="Proceed even if train/val frames failed to join a "
                         "label. Off by default -- an incomplete join is the "
                         "failure mode that quietly poisons every downstream "
                         "number.")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    frames_dir = Path(args.frames_dir)

    # ---- 1. every extracted frame on disk -------------------------------
    rows = []
    empty_dirs: list[str] = []
    canon_owner: dict[tuple[str, str], str] = {}
    for split_dir in sorted(p for p in frames_dir.iterdir() if p.is_dir()):
        for vid_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            jpgs = sorted(vid_dir.glob("*.jpg"))
            if not jpgs:
                empty_dirs.append(f"{split_dir.name}/{vid_dir.name}")
                continue
            # frame_path keeps the REAL on-disk directory name; only the join
            # key is canonicalised, so the loader still finds the file.
            vid = canonical_video_id(vid_dir.name)
            key = (split_dir.name, vid)
            if key in canon_owner and canon_owner[key] != vid_dir.name:
                raise SystemExit(
                    f"two frame directories collapse to the same video_id "
                    f"{vid!r} in {split_dir.name}: {canon_owner[key]!r} and "
                    f"{vid_dir.name!r}. Their frames would be interleaved into "
                    f"one fake video.")
            canon_owner[key] = vid_dir.name
            for j in jpgs:
                rows.append({
                    "frame_path": str(j.relative_to(frames_dir)),
                    "video_id": vid,
                    "frame_idx": int(j.stem),
                    "orig_split": split_dir.name,
                })
    frames = pd.DataFrame(rows)
    if frames.empty:
        raise SystemExit(f"No frames under {frames_dir}. Run extract_frames.py first.")
    log.info("on disk: %d frames, %d videos", len(frames), frames.video_id.nunique())

    if empty_dirs:
        log.error("%d extracted video folder(s) contain ZERO jpgs:",
                  len(empty_dirs))
        for d in empty_dirs[:10]:
            log.error("    %s", d)
        if len(empty_dirs) > 10:
            log.error("    ... and %d more", len(empty_dirs) - 10)
        if not args.allow_empty_dirs:
            raise SystemExit(
                f"{len(empty_dirs)} video folder(s) are empty, so those videos "
                f"are missing from the study entirely. On Windows this is "
                f"usually cv2.imwrite failing on a non-ASCII path and "
                f"returning False unchecked. Re-run extract_frames.py, or pass "
                f"--allow-empty-dirs to proceed without them.")

    # Reconcile against the extractor's own record. A video the extractor
    # counted but did not write is invisible to the disk scan above.
    manifest_path = frames_dir / "manifest.csv"
    if manifest_path.exists():
        man = pd.read_csv(manifest_path)
        man["video_id"] = man.video_id.map(canonical_video_id)
        on_disk = frames.groupby(["orig_split", "video_id"]).size()
        merged_man = man.set_index(["split", "video_id"]).join(
            on_disk.rename("actual"), how="left")
        merged_man["actual"] = merged_man["actual"].fillna(0).astype(int)
        bad = merged_man[merged_man.n_frames != merged_man.actual]
        if len(bad):
            log.error("manifest disagrees with disk for %d video(s); "
                      "%d frames recorded, %d present",
                      len(bad), int(bad.n_frames.sum()), int(bad.actual.sum()))
            if not args.allow_empty_dirs:
                raise SystemExit(
                    "manifest.csv and the extracted frames disagree. Re-run "
                    "extract_frames.py, or pass --allow-empty-dirs.")

    # ---- 2. every label CSV ---------------------------------------------
    # Scan them all. This dataset ships more than one label file and at least
    # one of them is empty, so a single file failing must not abort the run --
    # report per-file contribution and only give up if nothing yields labels.
    label_csvs = sorted(root.rglob("*.csv"))
    log.info("label files: %s", [str(p.relative_to(root)) for p in label_csvs])

    tables, contributions = [], []
    for p in label_csvs:
        try:
            t = load_label_table(p)
        except SystemExit as e:
            log.warning("  skipping %s: %s", p.relative_to(root), e)
            continue
        except Exception as e:
            log.warning("  skipping %s: %r", p.relative_to(root), e)
            continue
        n_pos = int((t.label == 1).sum()) if len(t) else 0
        contributions.append((str(p.relative_to(root)), len(t), n_pos))
        if len(t) == 0 or n_pos == 0:
            log.warning("  %s produced %d rows / %d positives -- ignoring it",
                        p.relative_to(root), len(t), n_pos)
            continue
        tables.append(t)

    log.info("per-file contribution:")
    for name, n, npos in contributions:
        log.info("    %-45s %7d rows  %7d positive", name, n, npos)

    if not tables:
        raise SystemExit(
            "No label file yielded any positive frames. Run "
            "the per-file contribution table above and check that the "
            "positive-index "
            "column name is listed in POS_INDEX_COLS in build_index.py.")

    labels = pd.concat(tables, ignore_index=True)
    if "is_seg_frame" not in labels.columns:
        labels["is_seg_frame"] = 0
    labels["is_seg_frame"] = labels["is_seg_frame"].fillna(0).astype(int)
    # Later files must not silently override earlier ones; keep the first
    # label seen for any (video, frame) pair and say how many collided.
    # DatasetV3 encodes val/test labels twice -- cls_label.csv and *_info.csv
    # carry the same frames. Deduplicating is fine ONLY while the copies agree;
    # "keep the first" applied to a genuine disagreement would pick a label by
    # file-scan order, which is arbitrary. So check before collapsing.
    key = ["video_id", "frame_idx"]
    dupes = int(labels.duplicated(subset=key).sum())
    if dupes:
        spread = labels.groupby(key).label.nunique()
        conflicts = spread[spread > 1]
        if len(conflicts):
            sample = conflicts.head(5).index.tolist()
            raise SystemExit(
                f"{len(conflicts)} (video, frame) pair(s) were given CONFLICTING "
                f"labels by different files, e.g. {sample}. Two label files "
                f"disagree about the same frame; resolve which is authoritative "
                f"rather than letting scan order decide.")
        log.info("%d (video, frame) pairs appear in more than one label file; "
                 "all copies agree, collapsing to one row each", dupes)
    labels = labels.drop_duplicates(subset=key, keep="first")
    log.info("labels: %d rows, %d videos, positive rate %.3f",
             len(labels), labels.video_id.nunique(), labels.label.mean())

    # ---- 3. join ---------------------------------------------------------
    merged = frames.merge(labels, on=["video_id", "frame_idx"], how="left")
    merged["is_seg_frame"] = merged["is_seg_frame"].fillna(0).astype(int)
    unlabelled = merged.label.isna()
    n_unlab = int(unlabelled.sum())
    merged.loc[unlabelled, "label"] = -1
    merged["label"] = merged["label"].astype(int)

    log.info("merged: %d frames, %d unlabelled (-1)", len(merged), n_unlab)
    if n_unlab:
        by_split = merged[unlabelled].orig_split.value_counts().to_dict()
        log.info("  unlabelled by original split: %s", by_split)

    # ---- 3b. the join must be complete ----------------------------------
    # This is an assertion, not a log line, because every failure found in this
    # pipeline so far has been silent: a sentinel that parsed to nothing, a
    # filename that did not match. Each one left the run alive and the numbers
    # wrong. DatasetV3 ships labels for all three folders, so a shortfall
    # anywhere means the join is broken.
    log.info("join rate by original split:")
    incomplete = []
    for s in sorted(merged.orig_split.unique()):
        sub = merged[merged.orig_split == s]
        rate = float((sub.label >= 0).mean())
        log.info("    %-12s %7d frames  join rate %.4f", s, len(sub), rate)
        if rate < 1.0:
            incomplete.append((s, len(sub), int((sub.label < 0).sum()), rate))

    if incomplete and not args.allow_unlabelled:
        detail = "; ".join(f"{s}: {miss}/{n} frames unlabelled (rate {r:.4f})"
                           for s, n, miss, r in incomplete)
        example = merged[merged.label < 0].video_id.drop_duplicates().head(5).tolist()
        raise SystemExit(
            f"incomplete label join -- {detail}. Example video_ids that found "
            f"no label: {example}. Check the per-file contribution table above "
            f"against the real CSVs; if a whole split is missing, the "
            f"video_id convention does not match (see canonical_video_id). "
            f"Pass --allow-unlabelled only if you have confirmed the shortfall "
            f"is genuine missing annotation.")

    lab = merged[merged.label >= 0]
    if lab.empty:
        raise SystemExit("Join produced zero labelled frames -- the video_id or "
                         "frame_idx conventions do not match. Inspect the "
                         "printed column names above against the real CSV.")

    log.info("USABLE: %d frames / %d videos, standard-plane rate %.4f",
             len(lab), lab.video_id.nunique(), lab.label.mean())

    # If labels only exist for one official split, our train/val/test all have
    # to be carved out of that split. Say so loudly -- it changes how many
    # videos the study actually has, and it belongs in the paper's methods.
    labelled_splits = sorted(lab.orig_split.unique())
    log.info("labelled data by original folder:")
    for s in labelled_splits:
        sub = lab[lab.orig_split == s]
        log.info("    %-12s %5d videos  %7d frames  pos-rate %.4f",
                 s, sub.video_id.nunique(), len(sub), sub.label.mean())

    if len(labelled_splits) == 1:
        log.warning("only the %r folder has usable labels. Your train/val/test "
                    "will all be drawn from those %d videos.",
                    labelled_splits[0], lab.video_id.nunique())
    else:
        log.info("NOTE: all %d official folders carry labels, including test/. "
                 "The earlier assumption that IUGC withholds the test labels "
                 "is not true of this Zenodo deposit, so the official split IS "
                 "usable. splits.py currently ignores "
                 "these folders and re-partitions every labelled video by "
                 "video id; that is a deliberate choice to revisit, not a "
                 "workaround for missing labels.", len(labelled_splits))

    # Annotation granularity differs by folder and it is not cosmetic: a train
    # video is labelled as a whole (ALL/NONE), so its frames carry a bag label,
    # while val/test frames are annotated individually. Report it here so the
    # distinction is visible at the point the index is built.
    gran = lab.groupby(["orig_split", "video_id"]).label.nunique()
    log.info("annotation granularity (videos whose frames are all one label):")
    for s in labelled_splits:
        sub = gran.loc[s]
        log.info("    %-12s %3d / %3d videos single-label  -> %s", s,
                 int((sub == 1).sum()), len(sub),
                 "video-level (weak) labels" if (sub == 1).all()
                 else "frame-level labels")

    per_vid = lab.groupby("video_id").agg(n=("label", "size"),
                                          pos=("label", "sum"))
    log.info("frames per video: min=%d median=%d max=%d",
             per_vid.n.min(), int(per_vid.n.median()), per_vid.n.max())
    log.info("videos with >=1 standard plane: %d / %d",
             int((per_vid.pos > 0).sum()), len(per_vid))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values(["video_id", "frame_idx"]).to_csv(out, index=False)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
