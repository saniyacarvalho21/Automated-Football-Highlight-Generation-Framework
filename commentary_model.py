# =========================================
# COMMENTARY EXCITEMENT SCORE [0, 1]
# =========================================
import subprocess, librosa, numpy as np, os, tempfile


def get_commentary_score(video_path):
    try:
        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vn", "-ac", "1", "-ar", "16000", tmp],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            return 0.0

        y, sr = librosa.load(tmp, sr=16000)
        os.remove(tmp)

        if len(y) == 0: return 0.0

        rms   = librosa.feature.rms(y=y)[0]
        delta = np.abs(np.diff(rms))

        # High energy + high variation = commentator excitement
        score = float(np.mean(rms)) + float(np.mean(delta))
        return float(np.clip(score * 5, 0.0, 1.0))

    except Exception:
        return 0.0