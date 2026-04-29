# =========================================
# ⚽ CINEMATIC PRO HIGHLIGHT GENERATOR
# Chronological + merged scenes + smooth FX
# =========================================

import os
import re
from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    vfx,
    TextClip,
    CompositeVideoClip
)

# -----------------------------------------
# Extract timestamp from seg_123.mp4
# -----------------------------------------
def get_time(path):
    name = os.path.basename(path)
    return int(re.findall(r"\d+", name)[0])


# -----------------------------------------
# Merge nearby clips into scenes
# Example:
# seg_10, seg_15, seg_20 → one 15s scene
# -----------------------------------------
def merge_segments(paths, gap=7):
    paths = sorted(paths, key=get_time)

    merged = []
    group = [paths[0]]

    for p in paths[1:]:
        if get_time(p) - get_time(group[-1]) <= gap:
            group.append(p)
        else:
            merged.append(group)
            group = [p]

    merged.append(group)
    return merged


# -----------------------------------------
# Add scoreboard overlay
# -----------------------------------------
def add_scoreboard(clip, text="⚽ HIGHLIGHTS"):
    title = TextClip(
        text,
        fontsize=40,
        color="white",
        font="Arial-Bold"
    ).set_position(("center", "top")).set_duration(clip.duration)

    return CompositeVideoClip([clip, title])


# -----------------------------------------
# MAIN GENERATOR
# -----------------------------------------
def generate(top_segments,
             out="data/outputs/highlight_final.mp4",
             max_minutes=6):

    if len(top_segments) == 0:
        print("❌ No segments")
        return

    # -------------------------------------------------
    # 1️⃣ Sort chronologically
    # -------------------------------------------------
    top_segments = sorted(top_segments, key=get_time)

    # -------------------------------------------------
    # 2️⃣ Merge nearby clips (cinematic scenes)
    # -------------------------------------------------
    scenes = merge_segments(top_segments, gap=7)

    clips = []
    total_time = 0
    max_seconds = max_minutes * 60

    # -------------------------------------------------
    # 3️⃣ Build scenes
    # -------------------------------------------------
    for group in scenes:

        scene_clips = [VideoFileClip(p) for p in group]
        scene = concatenate_videoclips(scene_clips, method="compose")

        # stop if too long
        if total_time + scene.duration > max_seconds:
            break

        # -------------------------------------------------
        # 4️⃣ Slow-motion for very exciting moments
        # (longer scene = likely goal)
        # -------------------------------------------------
        if scene.duration > 10:
            scene = scene.fx(vfx.speedx, 0.6)  # slow motion

        # -------------------------------------------------
        # 5️⃣ Smooth transitions
        # -------------------------------------------------
        scene = scene.fadein(0.4).fadeout(0.4)

        # -------------------------------------------------
        # 6️⃣ Scoreboard overlay
        # -------------------------------------------------
        scene = add_scoreboard(scene)

        clips.append(scene)
        total_time += scene.duration

    # -------------------------------------------------
    # 7️⃣ Final concatenate
    # -------------------------------------------------
    final = concatenate_videoclips(clips, method="compose")

    # -------------------------------------------------
    # 8️⃣ High quality export
    # -------------------------------------------------
    final.write_videofile(
        out,
        codec="libx264",
        audio_codec="aac",
        bitrate="6000k",
        preset="medium",
        threads=4,
        logger=None
    )

    print(f"✅ Cinematic highlight ready ({round(total_time/60,2)} min)")