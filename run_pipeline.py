import argparse
import os
import glob
import shutil
import subprocess
import sys
import traceback
import numpy as np
import pandas as pd

from src.segmenter import segment
from src.video_features import motion_score
from src.audio_features import audio_score
from src.yolo_ball import detect_ball_events
from src.commentary_model import get_commentary_score
from src.sis_model import predict_sis


SEG_DIR = "data/segments"
OUT_DIR = "data/outputs"
LOG_FILE = "data/outputs/pipeline.log"

# Full mode defaults
HIGHLIGHT_SEC = 360
SEG_LEN = 5

ALPHA, BETA, GAMMA = 0.45, 0.35, 0.20
W1, W2, W3 = 0.50, 0.30, 0.20
V1, V2, V3 = 0.40, 0.35, 0.25

SIS_THRESHOLD = 0.55
GOAL_AUDIO_THRESH = 0.50
GOAL_MOTION_THRESH = 0.35
PROXIMITY_WINDOW = 6
TEMPORAL_LAMBDA = 0.30


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def clean(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)


def normalize(series):
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-9:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - mn) / (mx - mn)


def temporal_weight(idx, total):
    return 1.0 + TEMPORAL_LAMBDA * (idx / total) if total else 1.0


def proximity_bonus(df, idx):
    nb = df[
        (df["segment_index"].between(idx - PROXIMITY_WINDOW, idx + PROXIMITY_WINDOW)) &
        (df["segment_index"] != idx)
    ]
    high = nb[(nb["audio_raw"] > 0.55) | (nb["motion_raw"] > 0.55)]
    return min(0.20, len(high) * 0.05)


def recency_decay(df, idx):
    nb = df[
        (df["segment_index"] >= idx - PROXIMITY_WINDOW) &
        (df["segment_index"] < idx)
    ]
    return max(0.5, 1.0 - len(nb) * 0.08)


def get_duration(path):
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ],
        capture_output=True,
        text=True
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("video", help="Path to input match video")
parser.add_argument("--fast", action="store_true", help="Use fast mode")
args = parser.parse_args()

video_path = args.video
FAST = args.fast

if FAST:
    SEG_LEN = 8
    HIGHLIGHT_SEC = 240

if not os.path.exists(video_path):
    print(f"File not found: {video_path}")
    sys.exit(1)

video_duration = get_duration(video_path)

clean(SEG_DIR)
clean(OUT_DIR)
open(LOG_FILE, "w").close()

log(f"Video : {video_path}")
log(f"Mode  : {'FAST' if FAST else 'FULL'}")
log(f"Size  : {os.path.getsize(video_path) / (1024 * 1024):.1f} MB")
log(f"Dur   : {video_duration / 60:.1f} min")
log(f"Seg   : {SEG_LEN} sec")
log(f"Est segments : {int(video_duration / SEG_LEN)}")

# --------------------------------------------------
# STAGE 1: SEGMENT
# --------------------------------------------------

log("\n[Stage 1/7] Segmenting video...")
try:
    segment(video_path, SEG_DIR, seg_len=SEG_LEN)
except TypeError:
    segment(video_path, SEG_DIR)
except Exception as e:
    log(f"ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

segment_files = sorted(glob.glob(f"{SEG_DIR}/*.mp4"))
if not segment_files:
    log("No segments created!")
    sys.exit(1)

total_segs = len(segment_files)
log(f"  ✅ {total_segs} segments")

# --------------------------------------------------
# STAGE 2: FEATURE EXTRACTION
# --------------------------------------------------

log(f"\n[Stage 2/7] Extracting features ({total_segs} segments)...")
rows = []

for i, f in enumerate(segment_files):
    try:
        idx = int(os.path.basename(f).split("_")[1].split(".")[0])
    except Exception:
        idx = i

    try:
        m = float(motion_score(f))
        a = float(audio_score(f))

        if FAST and (i % 3 != 0):
            yolo = 0.0
            comm = 0.0
        else:
            try:
                yolo = float(detect_ball_events(f))
            except Exception:
                yolo = 0.0

            try:
                comm = float(get_commentary_score(f))
            except Exception:
                comm = 0.0

    except Exception:
        m, a, yolo, comm = 0.0, 0.0, 0.0, 0.0

    rows.append({
        "segment_index": idx,
        "file": f,
        "audio_raw": a,
        "motion_raw": m,
        "yolo_raw": yolo,
        "commentary_raw": comm
    })

    if (i + 1) % max(1, total_segs // 10) == 0:
        pct = int((i + 1) / total_segs * 100)
        log(f"  {pct}% done ({i + 1}/{total_segs} segments)")

if not rows:
    log("No features extracted!")
    sys.exit(1)

df = pd.DataFrame(rows).sort_values("segment_index").reset_index(drop=True)

log(f"  ✅ {len(df)} segments processed")
log(f"  Audio  : {df['audio_raw'].min():.3f} – {df['audio_raw'].max():.3f}")
log(f"  Motion : {df['motion_raw'].min():.3f} – {df['motion_raw'].max():.3f}")

# --------------------------------------------------
# STAGE 3: NORMALIZATION + COMPONENTS
# --------------------------------------------------

log("\n[Stage 3/7] Building SIS components...")

df["motion_norm"] = normalize(df["motion_raw"])
df["density_score"] = normalize((df["motion_norm"] - df["motion_norm"].mean()).clip(lower=0))

df["audio_norm"] = normalize(df["audio_raw"])
df["audio_spike"] = df["audio_norm"].diff().fillna(0).abs()
df["replay_score"] = (df["audio_spike"] > 0.25).astype(float)

df["yolo_norm"] = normalize(df["yolo_raw"])
df["comm_norm"] = normalize(df["commentary_raw"])

df["V"] = W1 * df["motion_norm"] + W2 * df["density_score"] + W3 * df["replay_score"]

df["E"] = df["audio_norm"]
df["SF"] = normalize(df["audio_norm"].diff().abs().fillna(0))
df["P"] = normalize(df["audio_norm"].rolling(3, min_periods=1).std().fillna(0))
df["A"] = V1 * df["E"] + V2 * df["SF"] + V3 * df["P"]

c_list = []
for _, row in df.iterrows():
    idx = row["segment_index"]
    c_list.append(
        temporal_weight(idx, total_segs) *
        (1.0 + proximity_bonus(df, idx)) *
        recency_decay(df, idx)
    )

df["C_raw"] = c_list
df["C"] = normalize(pd.Series(df["C_raw"], index=df.index))

# --------------------------------------------------
# STAGE 4: SIS SCORING
# --------------------------------------------------

log("\n[Stage 4/7] SIS scoring...")

df["G"] = (
    (df["audio_norm"] > GOAL_AUDIO_THRESH) &
    (df["motion_norm"] > GOAL_MOTION_THRESH) &
    (df["density_score"] > 0.30)
).astype(float)

df["goal_like"] = (
    (df["yolo_norm"] > 0.5) &
    (df["audio_norm"] > 0.5)
).astype(float)

df["goal"] = df["G"].astype(int)
df["spike_bonus"] = df["audio_spike"].clip(upper=0.5)

use_model = not FAST

try:
    if use_model:
        features = df[["motion_norm", "audio_norm", "yolo_norm", "comm_norm", "audio_spike"]].values
        df["SIS"] = predict_sis(features)
        df["SIS"] = normalize(df["SIS"])
        log("  SIS model: OK")
    else:
        raise RuntimeError("Fast mode: skipping heavy SIS model")
except Exception as e:
    log(f"  SIS fallback: {e}")
    df["SIS"] = normalize(
        ALPHA * df["V"] +
        BETA * df["A"] +
        GAMMA * df["C"] +
        0.10 * df["G"] +
        0.05 * df["spike_bonus"]
    )

df["SIS"] = df["SIS"] + 0.25 * df["goal_like"] + 0.10 * df["yolo_norm"] + 0.08 * df["comm_norm"]
df["SIS"] = normalize(df["SIS"])

df["is_replay"] = ((df["audio_spike"] > 0.25) & (df["motion_norm"] < 0.4)).astype(int)
df["replay_group"] = (df["is_replay"].diff() != 0).cumsum()

for _, grp in df[df["is_replay"] == 1].groupby("replay_group"):
    if len(grp) > 2:
        df.loc[grp.index[2:], "SIS"] *= 0.5

df["SIS"] = normalize(df["SIS"])

log(f"  SIS range : {df['SIS'].min():.3f} – {df['SIS'].max():.3f}")
log(f"  Goal segs : {int(df['G'].sum())}")

# --------------------------------------------------
# STAGE 5: SELECT HIGHLIGHTS
# --------------------------------------------------

log("\n[Stage 5/7] Selecting highlights...")

top_needed = max(1, HIGHLIGHT_SEC // SEG_LEN)

highlight_df = df[(df["SIS"] >= SIS_THRESHOLD) & (df["is_replay"] == 0)].copy()
goal_segs = df[df["goal_like"] == 1].copy()

highlight_df = pd.concat([highlight_df, goal_segs]).drop_duplicates("segment_index")

if len(highlight_df) < top_needed:
    for tau in [0.45, 0.35, 0.25, 0.15, 0.0]:
        extra = df[
            (df["SIS"] >= tau) &
            (~df["segment_index"].isin(highlight_df["segment_index"]))
        ]
        highlight_df = pd.concat([highlight_df, extra]).drop_duplicates("segment_index")
        log(f"  tau={tau} → {len(highlight_df)} segments")
        if len(highlight_df) >= top_needed:
            break

if highlight_df.empty:
    highlight_df = df.copy()

top_df = (
    highlight_df
    .sort_values("SIS", ascending=False)
    .head(top_needed)
    .sort_values("segment_index")
    .reset_index(drop=True)
)

dur_est = len(top_df) * SEG_LEN
log(f"  ✅ {len(top_df)} segments selected (~{dur_est // 60}m {dur_est % 60}s)")

np.save(f"{OUT_DIR}/scores.npy", df["SIS"].values)
np.save(f"{OUT_DIR}/energy.npy", df["A"].values)
top_df.to_csv(f"{OUT_DIR}/top_moments.csv", index=False)
df.to_csv(f"{OUT_DIR}/full_analysis.csv", index=False)
log("  Analytics saved")

# --------------------------------------------------
# STAGE 6: ASSEMBLY
# --------------------------------------------------

log("\n[Stage 6/7] Building highlight via FFmpeg...")

final_output = f"{OUT_DIR}/highlight.mp4"
filter_parts = []
input_labels_v = []
input_labels_a = []

for i, (_, row) in enumerate(top_df.iterrows()):
    start_sec = int(row["segment_index"]) * SEG_LEN
    end_sec = min(start_sec + SEG_LEN, video_duration)
    if end_sec <= start_sec:
        continue

    filter_parts.append(
        f"[0:v]trim=start={start_sec}:end={end_sec},"
        f"setpts=PTS-STARTPTS,"
        f"scale=1280:720:force_original_aspect_ratio=decrease,"
        f"pad=1280:720:(ow-iw)/2:(oh-ih)/2[v{i}];"
    )

    filter_parts.append(
        f"[0:a]atrim=start={start_sec}:end={end_sec},"
        f"asetpts=PTS-STARTPTS[a{i}];"
    )

    input_labels_v.append(f"[v{i}]")
    input_labels_a.append(f"[a{i}]")

if not input_labels_v:
    log("No clips to assemble!")
    sys.exit(1)

n = len(input_labels_v)
concat_str = "".join(input_labels_v) + "".join(input_labels_a)
filter_parts.append(f"{concat_str}concat=n={n}:v=1:a=1[outv][outa]")

filter_file = f"{OUT_DIR}/filter.txt"
with open(filter_file, "w", encoding="utf-8") as ff:
    ff.write("\n".join(filter_parts))

preset = "veryfast" if FAST else "fast"
crf = "24" if FAST else "23"

log(f"  Merging {n} clips...")

result = subprocess.run([
    "ffmpeg", "-y",
    "-i", video_path,
    "-filter_complex_script", filter_file,
    "-map", "[outv]", "-map", "[outa]",
    "-c:v", "libx264", "-preset", preset, "-crf", crf,
    "-c:a", "aac", "-b:a", "128k",
    final_output
], capture_output=True, text=True)

if result.returncode != 0:
    log("  filter_complex failed — trying concat fallback...")
    concat_list = f"{OUT_DIR}/concat.txt"

    with open(concat_list, "w", encoding="utf-8") as cf:
        for _, row in top_df.iterrows():
            if os.path.exists(row["file"]):
                cf.write(f"file '{os.path.abspath(row['file'])}'\n")

    r2 = subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c:v", "libx264", "-preset", preset, "-crf", crf,
        "-c:a", "aac",
        final_output
    ], capture_output=True, text=True)

    if r2.returncode != 0:
        log(f"  Fallback error: {r2.stderr[-1000:]}")
        sys.exit(1)

# --------------------------------------------------
# STAGE 7: DONE
# --------------------------------------------------

log("\n[Stage 7/7] Done!")

if os.path.exists(final_output) and os.path.getsize(final_output) > 0:
    size = os.path.getsize(final_output) / (1024 * 1024)
    log(f"✅ HIGHLIGHT READY : {final_output}")
    log(f"   Duration : ~{dur_est // 60}m {dur_est % 60}s")
    log(f"   Size     : {size:.1f} MB")
    log(f"   Goals    : {int(top_df['G'].sum()) if 'G' in top_df.columns else '?'}")
else:
    log("❌ FAILED — highlight.mp4 not created")
    sys.exit(1)