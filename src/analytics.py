# ==========================================
# ⚽ FOOTBALL HIGHLIGHT PIPELINE — FINAL STABLE VERSION
# FAST + RELIABLE + SCALABLE 🚀
# ==========================================

import os, glob, shutil, subprocess, sys
import numpy as np
import pandas as pd

from src.video_features import motion_score
from src.audio_features import audio_score
from src.yolo_ball import detect_ball_events
from src.commentary_model import get_commentary_score
from src.sis_model import predict_sis


# -----------------------------
# CONFIG
# -----------------------------
SEG_DIR = "data/segments"
OUT_DIR = "data/outputs"
LIST_FILE = f"{OUT_DIR}/concat_list.txt"

SIS_THRESHOLD = 0.55

ALPHA, BETA, GAMMA = 0.45, 0.35, 0.20


# -----------------------------
# CLEAN UTILS
# -----------------------------
def clean(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)


def normalize(x):
    x = np.array(x, dtype=float)
    if len(x) == 0:
        return x
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


# -----------------------------
# INPUT
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python run_pipeline.py <video>")
    sys.exit(1)

video_path = sys.argv[1]

print("\n🎬 Processing:", video_path)

clean(OUT_DIR)


# -----------------------------
# LOAD SEGMENTS (ASSUMED PRE-SMART SEGMENTATION)
# -----------------------------
print("\n[1] Loading segments...")

files = sorted(glob.glob(f"{SEG_DIR}/*.mp4"))

if not files:
    print("❌ No segments found. Run segmentation first.")
    sys.exit(1)

print("✅ Segments loaded:", len(files))


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
print("\n[2] Extracting features...")

rows = []

for f in files:
    try:
        idx = int(os.path.basename(f).split("_")[1].split(".")[0])
    except:
        idx = 0

    try:
        motion = motion_score(f)
        audio = audio_score(f)
        yolo = detect_ball_events(f)
        comm = get_commentary_score(f)
    except:
        motion, audio, yolo, comm = 0, 0, 0, 0

    rows.append({
        "segment_index": idx,
        "file": f,
        "motion": motion,
        "audio": audio,
        "yolo": yolo,
        "comm": comm
    })


df = pd.DataFrame(rows)


# -----------------------------
# NORMALIZATION
# -----------------------------
print("\n[3] Normalizing...")

df["motion_n"] = normalize(df["motion"])
df["audio_n"]  = normalize(df["audio"])
df["yolo_n"]   = normalize(df["yolo"])
df["comm_n"]   = normalize(df["comm"])

df["audio_spike"] = np.abs(np.diff(df["audio_n"], prepend=0))


# -----------------------------
# SIS SCORING
# -----------------------------
print("\n[4] SIS scoring...")

V = 0.5*df["motion_n"] + 0.3*df["yolo_n"] + 0.2*df["audio_spike"]
A = 0.6*df["audio_n"] + 0.4*df["audio_spike"]
C = df["segment_index"] / max(len(df), 1)

try:
    df["SIS"] = predict_sis(
        df[["motion_n","audio_n","yolo_n","comm_n","audio_spike"]].values
    )
except:
    df["SIS"] = ALPHA*V + BETA*A + GAMMA*C

df["SIS"] = normalize(df["SIS"])


# -----------------------------
# GOAL BOOST (FAST RULE)
# -----------------------------
df["goal"] = (
    (df["yolo_n"] > 0.5) &
    (df["audio_n"] > 0.6) &
    (df["audio_spike"] > 0.3)
)

df.loc[df["goal"], "SIS"] += 0.25
df["SIS"] = normalize(df["SIS"])


# -----------------------------
# REPLAY FILTER
# -----------------------------
df["is_replay"] = (df["audio_spike"] > 0.25) & (df["motion_n"] < 0.4)


# -----------------------------
# 🎯 SELECTION
# -----------------------------
print("\n[5] Selecting highlights...")

selected = df[(df["SIS"] > SIS_THRESHOLD) & (~df["is_replay"])]

# fallback guarantee
if len(selected) < 60:
    selected = df.sort_values("SIS", ascending=False).head(60)

selected = selected.sort_values("segment_index")


# -----------------------------
# 🎬 BUILD CONCAT LIST
# -----------------------------
print("\n[6] Building concat file...")

os.makedirs(OUT_DIR, exist_ok=True)

with open(LIST_FILE, "w") as f:
    for file in selected["file"]:
        f.write(f"file '{os.path.abspath(file)}'\n")


output = f"{OUT_DIR}/highlight.mp4"


# -----------------------------
# 🎬 FAST RENDER (STABLE FFmpeg COPY MODE)
# -----------------------------
print("\n[7] Rendering video...")

subprocess.run([
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", LIST_FILE,
    "-c", "copy",
    output
])


# -----------------------------
# DONE
# -----------------------------
if os.path.exists(output):
    print("\n🎉 HIGHLIGHT READY:", output)
else:
    print("❌ FAILED TO GENERATE VIDEO")