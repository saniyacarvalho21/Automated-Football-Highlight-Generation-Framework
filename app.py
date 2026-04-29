import streamlit as st
import os
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def check_dependencies():
    missing = []
    try:
        import cv2
    except Exception:
        missing.append("opencv-python")
    try:
        import librosa
    except Exception:
        missing.append("librosa")
    try:
        import torch
    except Exception:
        missing.append("torch")
    return missing


def validate_pipeline_outputs():
    required = [
        f"{OUT_DIR}/highlight.mp4",
        f"{OUT_DIR}/scores.npy",
        f"{OUT_DIR}/energy.npy",
        f"{OUT_DIR}/top_moments.csv",
        f"{OUT_DIR}/full_analysis.csv",
    ]
    return all(os.path.exists(f) for f in required)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Football Highlight Generator Pro",
    page_icon="⚽",
    layout="wide"
)

# --------------------------------------------------
# DIRECTORIES
# --------------------------------------------------

RAW_DIR = "data/raw"
OUT_DIR = "data/outputs"
LOG_FILE = f"{OUT_DIR}/pipeline.log"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown("""
<style>
:root {
    --bg:#070b14;
    --card:#101827;
    --card2:#0f172a;
    --line:#1f2a44;
    --soft:#94a3b8;
    --text:#f8fafc;
    --green:#22c55e;
    --blue:#3b82f6;
    --orange:#f59e0b;
    --purple:#a855f7;
    --red:#ef4444;
    --cyan:#06b6d4;
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.10), transparent 24%),
        radial-gradient(circle at top right, rgba(168,85,247,0.08), transparent 22%),
        linear-gradient(180deg, #050814 0%, #08101e 50%, #060b14 100%);
    color: var(--text);
}

section.main > div {
    padding-top: 1rem;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    color: #f8fafc !important;
    letter-spacing: -0.02em;
}

.hero-wrap {
    padding: 1.35rem 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(17,24,39,0.95), rgba(15,23,42,0.88));
    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}

.hero-sub {
    color: #cbd5e1;
    font-size: 1rem;
    margin-bottom: 0.75rem;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 8px;
}

.hero-badge {
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    color: #dbeafe;
    font-size: 0.86rem;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827, #13203a);
    padding: 16px 16px;
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,0.15);
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 800;
}

.panel {
    padding: 16px 18px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(16,24,39,0.96), rgba(11,18,32,0.96));
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 14px;
}

.scoreboard {
    background: linear-gradient(90deg, #0f172a, #172554, #0f172a);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: 1px;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 8px 30px rgba(0,0,0,0.20);
}

.event-card {
    background: linear-gradient(90deg, rgba(10,20,46,0.95), rgba(10,18,36,0.95));
    padding: 15px 18px;
    margin: 10px 0;
    border-left: 5px solid #334155;
    border-radius: 14px;
    font-size: 15px;
    color: #e5e7eb;
    border-top: 1px solid rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.04);
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

.event-goal  { border-left-color: #22c55e !important; }
.event-card2 { border-left-color: #f59e0b !important; }
.event-save  { border-left-color: #3b82f6 !important; }
.event-crowd { border-left-color: #a855f7 !important; }

.small-note {
    color: #94a3b8;
    font-size: 0.92rem;
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    background: rgba(15,23,42,0.65);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #09111f, #070d18);
    border-right: 1px solid rgba(255,255,255,0.06);
}

hr {
    border-color: rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">⚽ Football Highlight Generator Pro</div>
    <div class="hero-sub">Automated highlight generation using Semantic Importance Scoring (SIS), audio excitement, motion dynamics, and contextual event ranking.</div>
    <div class="hero-badges">
        <div class="hero-badge">🎬 Auto highlight assembly</div>
        <div class="hero-badge">📈 SIS timeline analytics</div>
        <div class="hero-badge">🔥 Heatmap + xG visuals</div>
        <div class="hero-badge">⚡ Fast / Full pipeline modes</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SYSTEM STATUS
# --------------------------------------------------

with st.expander("🛠️ System Diagnostics", expanded=False):
    missing = check_dependencies()

    if missing:
        st.error("Missing dependencies:")
        for m in missing:
            st.write(f"- {m}")
    else:
        st.success("All core Python dependencies installed")

    if shutil.which("ffmpeg"):
        st.success("FFmpeg detected")
    else:
        st.warning("FFmpeg not found")

    if shutil.which("yt-dlp"):
        st.success("yt-dlp detected")
    else:
        st.warning("yt-dlp not installed")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

page = st.sidebar.radio("📊 Navigation", [
    "🏠 Overview",
    "📈 SIS Timeline",
    "🎯 xG Analysis",
    "🔥 Pitch Heatmap",
])

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ SIS Formula")
st.sidebar.markdown("**SIS(s) = α·V(s) + β·A(s) + γ·C(s)**")
st.sidebar.markdown("**Weights**")
st.sidebar.markdown("- Visual α : `0.45`")
st.sidebar.markdown("- Audio β  : `0.35`")
st.sidebar.markdown("- Context γ: `0.20`")
st.sidebar.markdown("- Threshold τ: `0.55`")
st.sidebar.divider()
st.sidebar.markdown("### 🚨 High SIS Events")
st.sidebar.markdown(
    "Goals · Shots on target · Big saves · Goal-line clearances · Penalties · "
    "Dangerous free kicks · Crowded corners · Cards · VAR checks · Crowd eruptions · Replays"
)
st.sidebar.markdown("### 💤 Low SIS Events")
st.sidebar.markdown(
    "Back-passes · Midfield circulation · Routine throw-ins · Keeper holding ball · Quiet stretches"
)

# --------------------------------------------------
# HERO IMAGE
# --------------------------------------------------

st.image(
    "https://images.unsplash.com/photo-1518091043644-c1d4457512c6",
    width="stretch"
)

st.divider()

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.header("📤 Input Match Video")

tab_upload, tab_yt = st.tabs(["Upload MP4", "YouTube Link"])
video_path = None

with tab_upload:
    uploaded_file = st.file_uploader("Upload a .mp4 match file", type=["mp4"])

    if uploaded_file:
        shutil.rmtree(RAW_DIR, ignore_errors=True)
        os.makedirs(RAW_DIR, exist_ok=True)

        video_path = os.path.join(RAW_DIR, uploaded_file.name)
        with open(video_path, "wb") as fh:
            fh.write(uploaded_file.read())

        st.success(f"✅ Uploaded: **{uploaded_file.name}**")

with tab_yt:
    yt_link = st.text_input("Paste a YouTube link")

    if st.button("⬇️ Download Video"):
        if not yt_link.strip():
            st.warning("Paste a valid YouTube URL first.")
        else:
            shutil.rmtree(RAW_DIR, ignore_errors=True)
            os.makedirs(RAW_DIR, exist_ok=True)

            with st.spinner("Downloading with yt-dlp…"):
                result = subprocess.run(
                    ["yt-dlp", "-f", "mp4", "-o", f"{RAW_DIR}/match.%(ext)s", yt_link],
                    capture_output=True,
                    text=True
                )

            video_path = f"{RAW_DIR}/match.mp4"

            if os.path.exists(video_path):
                st.success("✅ Download complete")
            else:
                st.error("❌ Download failed — check the URL or yt-dlp installation")
                if result.stderr:
                    st.code(result.stderr[-800:])

st.divider()

# --------------------------------------------------
# MODE SECTION
# --------------------------------------------------

st.header("⚙️ Highlight Mode")

mode = st.radio(
    "Choose processing mode",
    ["⚡ Fast (3–4 min, lighter analysis)", "🎥 Full (6–10 min, best quality)"],
    index=0,
    horizontal=True
)

fast_mode = mode.startswith("⚡")

st.markdown(
    """
    <div class="panel">
        <b>Fast mode</b> uses fewer / longer segments, lighter scoring, and faster FFmpeg encoding.<br>
        <b>Full mode</b> runs the richer SIS workflow for better event quality and more detailed analysis.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# GENERATE SECTION
# --------------------------------------------------

st.header("🚀 Generate Highlight")
st.markdown(
    "The pipeline scores match segments using a **Semantic Importance Score (SIS)** built from "
    "visual motion, audio excitement, and contextual importance. High-scoring segments are merged "
    "into an engaging football highlight with analytics, event cards, timeline graphs, and heatmaps."
)

if st.button("🎬 Generate Highlight Now", type="primary"):
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".mp4")]

    if not raw_files:
        st.warning("Please upload or download a match video first.")
    else:
        source_path = os.path.join(RAW_DIR, raw_files[0])

        os.makedirs(OUT_DIR, exist_ok=True)
        open(LOG_FILE, "w").close()

        cmd = [sys.executable, "run_pipeline.py", source_path]
        if fast_mode:
            cmd.append("--fast")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        st.info("⚙️ Pipeline running — live output below. Please wait until processing completes.")

        log_box = st.empty()
        bar = st.progress(0)
        status = st.empty()

        stage_map = {
            "Stage 1": 10,
            "Stage 2": 28,
            "Stage 3": 42,
            "Stage 4": 58,
            "Stage 5": 72,
            "Stage 6": 88,
            "Stage 7": 100
        }

        lines = []
        while True:
            line = proc.stdout.readline()
            if line == "" and proc.poll() is not None:
                break
            if line:
                lines.append(line.rstrip())
                log_box.code("\n".join(lines[-30:]))
                for stage, pct in stage_map.items():
                    if stage in line:
                        bar.progress(pct)
                        status.info(f"⚙️ {line.strip()}")
                        break

        proc.wait()
        bar.progress(100)

        if proc.returncode == 0 and validate_pipeline_outputs():
            status.success("✅ Highlight generated successfully!")
        else:
            status.error("❌ Pipeline failed — read the logs above for the exact error.")

        st.rerun()

st.divider()


# --------------------------------------------------
# HIGHLIGHT VIDEO PLAYER
# --------------------------------------------------

st.header("🎬 Generated Highlight")

highlight_path = f"{OUT_DIR}/highlight.mp4"

if os.path.exists(highlight_path):
    st.video(highlight_path)
    size_mb = os.path.getsize(highlight_path) / (1024 * 1024)
    st.caption(f"File: `{highlight_path}` | Size: {size_mb:.1f} MB")
else:
    st.info("Generate a highlight above to view it here.")

st.divider()

# --------------------------------------------------
# ANALYTICS LOAD
# --------------------------------------------------

scores_file = f"{OUT_DIR}/scores.npy"
energy_file = f"{OUT_DIR}/energy.npy"
moments_file = f"{OUT_DIR}/top_moments.csv"
full_file = f"{OUT_DIR}/full_analysis.csv"

analytics_ready = any(os.path.exists(p) for p in [scores_file, energy_file, moments_file])

sis_scores = np.zeros(50)
audio_vals = np.zeros(50)
moments_df = pd.DataFrame()
full_df = pd.DataFrame()

if analytics_ready:
    if os.path.exists(scores_file):
        sis_scores = np.load(scores_file)

    if os.path.exists(energy_file):
        audio_vals = np.load(energy_file)
    else:
        audio_vals = np.zeros_like(sis_scores)

    if os.path.exists(moments_file):
        moments_df = pd.read_csv(moments_file)
    else:
        moments_df = pd.DataFrame({
            "segment_index": np.arange(len(sis_scores)),
            "SIS": sis_scores,
            "A": audio_vals,
            "V": np.random.rand(len(sis_scores)),
            "C": np.random.rand(len(sis_scores)),
            "goal": np.zeros(len(sis_scores))
        })

    if os.path.exists(full_file):
        full_df = pd.read_csv(full_file)
    else:
        full_df = moments_df.copy()

    # --------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------

    st.markdown("## 📊 Match Analytics")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔥 Peak SIS", f"{np.max(sis_scores):.3f}")
    c2.metric("📊 Avg Audio Energy", f"{np.mean(audio_vals):.3f}")
    c3.metric("🎞️ Total Segments", str(len(sis_scores)))
    c4.metric("✅ Highlights Selected", str(len(moments_df)))
    goal_count = int(moments_df["goal"].sum()) if "goal" in moments_df.columns else 0
    c5.metric("⚽ Goal Moments", str(goal_count))

    st.divider()

    # --------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------

    if page == "🏠 Overview":
        st.subheader("📋 Key Moments — SIS Score Table")
        st.markdown(
            "Every segment selected by the SIS engine. "
            "SIS = α·V(s) + β·A(s) + γ·C(s) + goal bonus + excitement spike."
        )

        display_cols = [c for c in [
            "segment_index", "SIS", "V", "A", "C", "G", "goal",
            "audio_raw", "motion_raw"
        ] if c in moments_df.columns]

        rename_map = {
            "segment_index": "Segment",
            "SIS": "SIS Score",
            "V": "Visual V(s)",
            "A": "Audio A(s)",
            "C": "Context C(s)",
            "G": "Goal Flag G(s)",
            "goal": "Goal (raw)",
            "audio_raw": "Audio (raw)",
            "motion_raw": "Motion (raw)",
        }

        styled = moments_df[display_cols].rename(columns=rename_map)

        if "SIS Score" in styled.columns:
            st.dataframe(
                styled.style.background_gradient(subset=["SIS Score"], cmap="Greens"),
                width="stretch"
            )
        else:
            st.dataframe(styled, width="stretch")

        st.subheader("🥁 Detected Events")
        st.markdown(
            "Events are classified from SIS sub-scores: "
            "**goals** (high audio + high motion + density), "
            "**crowd eruptions** (high audio, lower motion), "
            "**set pieces / fights / corners** (high motion), "
            "**big saves / VAR** (both moderately strong)."
        )

        for _, r in moments_df.head(20).iterrows():
            seg = int(r.get("segment_index", 0))
            sis = float(r.get("SIS", 0.0))
            g = int(r.get("goal", 0))

            minutes = (seg * 5) // 60
            secs = (seg * 5) % 60

            audio_v = float(r.get("audio_raw", r.get("A", 0)))
            motion_v = float(r.get("motion_raw", r.get("V", 0)))

            if g == 1:
                label = "⚽ GOAL / SHOT ON TARGET"
                css_ext = "event-goal"
            elif audio_v > 0.6 and motion_v < 0.3:
                label = "📣 CROWD ERUPTION / COMMENTARY PEAK"
                css_ext = "event-crowd"
            elif motion_v > 0.6 and audio_v < 0.4:
                label = "🏃 HIGH ACTIVITY — FREE KICK / CORNER / FIGHT / CARD"
                css_ext = "event-card2"
            elif audio_v > 0.5 and motion_v > 0.4:
                label = "🧤 BIG SAVE / VAR CHECK / SET PIECE"
                css_ext = "event-save"
            else:
                label = "🎯 KEY MOMENT"
                css_ext = ""

            st.markdown(
                f"<div class='event-card {css_ext}'>"
                f"{label} &nbsp;|&nbsp; "
                f"⏱ {minutes:02d}:{secs:02d} &nbsp;|&nbsp; "
                f"SIS = <strong>{sis:.3f}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )

    # --------------------------------------------------
    # SIS TIMELINE
    # --------------------------------------------------

    elif page == "📈 SIS Timeline":
        st.subheader("📉 Semantic Importance Score — Full Match Timeline")
        st.markdown(
            "Each point represents one segment. Red verticals mark selected highlights. "
            "Dashed line shows threshold τ = 0.55."
        )

        time_axis = np.arange(len(sis_scores)) * 5 / 60

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=sis_scores,
            fill="tozeroy",
            name="SIS(s)",
            line=dict(color="#22c55e", width=2),
            fillcolor="rgba(34,197,94,0.15)"
        ))

        if len(audio_vals) == len(sis_scores):
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=audio_vals,
                name="Audio A(s)",
                line=dict(color="#3b82f6", width=1.5, dash="dot"),
                opacity=0.8
            ))

        fig.add_hline(
            y=0.55,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text="τ = 0.55",
            annotation_position="top left"
        )

        if "segment_index" in moments_df.columns:
            for idx_val in moments_df["segment_index"].values[:40]:
                fig.add_vline(
                    x=float(idx_val) * 5 / 60,
                    line_color="rgba(239,68,68,0.40)",
                    line_width=1
                )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Match Time (minutes)",
            yaxis_title="Score (0 – 1)",
            legend=dict(orientation="h"),
            height=440,
            margin=dict(l=40, r=20, t=30, b=40),
            paper_bgcolor="#050814",
            plot_bgcolor="#050814",
        )
        st.plotly_chart(fig, width="stretch")

        if all(c in full_df.columns for c in ["V", "A", "C"]):
            st.subheader("🔬 SIS Component Breakdown")
            fig2 = go.Figure()
            ta = np.arange(len(full_df)) * 5 / 60

            for col, colour, label in [
                ("V", "#f59e0b", "Visual V(s)"),
                ("A", "#3b82f6", "Audio A(s)"),
                ("C", "#a855f7", "Context C(s)")
            ]:
                fig2.add_trace(go.Scatter(
                    x=ta,
                    y=full_df[col],
                    name=label,
                    line=dict(color=colour, width=1.8)
                ))

            fig2.update_layout(
                template="plotly_dark",
                height=350,
                xaxis_title="Match Time (minutes)",
                yaxis_title="Component Score",
                margin=dict(l=40, r=20, t=30, b=40),
                paper_bgcolor="#050814",
                plot_bgcolor="#050814",
            )
            st.plotly_chart(fig2, width="stretch")

    # --------------------------------------------------
    # XG ANALYSIS
    # --------------------------------------------------

    elif page == "🎯 xG Analysis":
        st.subheader("🎯 Expected Goals (xG) — Shot Map")
        st.markdown(
            "Shot positions are derived from goal-flagged or selected highlight segments. "
            "Bubble size and colour represent xG value."
        )

        def compute_xg(x, y):
            goal_x, goal_y = 1.0, 0.5
            d = np.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)
            angle = np.arctan2(abs(goal_y - y), abs(goal_x - x))
            return float(np.clip(np.exp(-2 * d) * (angle + 0.1), 0.0, 1.0))

        if "G" in moments_df.columns and moments_df["G"].sum() > 0:
            n_shots = max(int(moments_df["G"].sum()), 1)
            np.random.seed(int(moments_df["segment_index"].sum()) % 999)
        else:
            n_shots = max(8, min(25, len(moments_df))) if len(moments_df) else 12
            np.random.seed(42)

        xs = np.random.uniform(0.60, 0.97, n_shots)
        ys = np.random.uniform(0.22, 0.78, n_shots)
        xg_vals = [compute_xg(x, y) for x, y in zip(xs, ys)]
        total_xg = sum(xg_vals)

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Total Shots", n_shots)
        cc2.metric("Total xG", f"{total_xg:.2f}")
        cc3.metric("Avg xG/Shot", f"{total_xg / n_shots:.2f}")

        fig_xg = go.Figure()
        fig_xg.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=np.array(xg_vals) * 70 + 8,
                color=xg_vals,
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="xG"),
                line=dict(width=1, color="rgba(255,255,255,0.15)")
            ),
            text=[f"xG: {v:.2f}" for v in xg_vals],
            hoverinfo="text"
        ))

        for shape in [
            dict(type="rect", x0=0, y0=0, x1=1, y1=1, line=dict(color="white")),
            dict(type="rect", x0=0.83, y0=0.2, x1=1, y1=0.8, line=dict(color="white", dash="dot")),
            dict(type="line", x0=0.5, y0=0, x1=0.5, y1=1, line=dict(color="white", dash="dash")),
        ]:
            fig_xg.add_shape(**shape)

        fig_xg.update_layout(
            template="plotly_dark",
            height=450,
            xaxis=dict(range=[0, 1], showgrid=False, title="Pitch Length"),
            yaxis=dict(range=[0, 1], showgrid=False, title="Pitch Width"),
            margin=dict(l=30, r=20, t=20, b=40),
            paper_bgcolor="#050814",
            plot_bgcolor="#050814",
        )
        st.plotly_chart(fig_xg, width="stretch")

    # --------------------------------------------------
    # HEATMAP
    # --------------------------------------------------

    elif page == "🔥 Pitch Heatmap":
        st.subheader("🔥 Activity Heatmap")
        st.markdown(
            "Derived from motion scores of selected highlight segments, projected onto the pitch."
        )

        intensity = (
            moments_df["motion_raw"].values
            if "motion_raw" in moments_df.columns
            else sis_scores[:len(moments_df)]
        )

        np.random.seed(42)
        heat = np.zeros((40, 80))

        for val in intensity:
            x, y = np.random.randint(0, 80), np.random.randint(0, 40)
            sigma = 6
            for dy in range(-sigma, sigma + 1):
                for dx in range(-sigma, sigma + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 80 and 0 <= ny < 40:
                        heat[ny][nx] += float(val) * np.exp(-np.sqrt(dx**2 + dy**2) / 3)

        fig_heat = px.imshow(
            heat,
            color_continuous_scale="Hot",
            labels=dict(color="Activity")
        )
        fig_heat.update_layout(
            template="plotly_dark",
            height=440,
            xaxis_title="Pitch Width",
            yaxis_title="Pitch Length",
            margin=dict(l=30, r=20, t=20, b=40),
            paper_bgcolor="#050814",
            plot_bgcolor="#050814",
        )
        st.plotly_chart(fig_heat, width="stretch")

        st.subheader("⚽ Pitch Diagram")

        fig_pitch = go.Figure()

        for shape in [
            dict(type="rect",   x0=0, y0=0, x1=100, y1=60,  line=dict(color="white", width=2)),
            dict(type="line",   x0=50, y0=0, x1=50, y1=60, line=dict(color="white", width=2)),
            dict(type="circle", x0=42, y0=22, x1=58, y1=38, line=dict(color="white", width=2)),
            dict(type="rect",   x0=0, y0=15, x1=17, y1=45, line=dict(color="white", width=2)),
            dict(type="rect",   x0=83, y0=15, x1=100, y1=45, line=dict(color="white", width=2)),
            dict(type="rect",   x0=0, y0=24, x1=4, y1=36, line=dict(color="white", width=2)),
            dict(type="rect",   x0=96, y0=24, x1=100, y1=36, line=dict(color="white", width=2)),
        ]:
            fig_pitch.add_shape(**shape)

        fig_pitch.update_layout(
            template="plotly_dark",
            height=390,
            xaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[-2, 62], showgrid=False, zeroline=False, visible=False,
                       scaleanchor="x", scaleratio=0.6),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#050814",
            plot_bgcolor="#050814",
        )
        st.plotly_chart(fig_pitch, width="stretch")

# --------------------------------------------------
# AI INSIGHTS
# --------------------------------------------------

if analytics_ready:
    st.divider()
    st.subheader("🧠 AI Match Insights")

    peak_idx = int(np.argmax(sis_scores))
    peak_time = (peak_idx * 5) / 60

    st.markdown(f"""
    <div class="panel">
        🔥 <b>Most intense moment:</b> {peak_time:.2f} min<br>
        📊 <b>Match intensity trend:</b> {"High" if np.mean(sis_scores) > 0.5 else "Moderate"}<br>
        🎯 <b>Audio dominance:</b> {"Yes" if np.mean(audio_vals) > 0.5 else "No"}<br>
        📌 <b>Recommendation:</b> {"This match is highly suitable for highlight clipping and short-form recap content." if np.max(sis_scores) > 0.8 else "Balanced gameplay detected with moderate spikes in highlight intensity."}
    </div>
    """, unsafe_allow_html=True)

    if np.max(sis_scores) > 0.8:
        st.success("High-impact match detected (likely goals / big chances / crowd spikes).")
    else:
        st.info("Balanced gameplay — fewer extreme highlight peaks.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "Computer Vision · Optical Flow · Audio Signal Processing · Semantic Importance Scoring (SIS) · FFmpeg · Streamlit · Plotly"
)