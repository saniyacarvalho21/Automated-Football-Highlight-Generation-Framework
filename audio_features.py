# =========================================
# AUDIO SCORE — normalized [0, 1]
# =========================================
import subprocess, librosa, numpy as np, os, tempfile


def extract_audio_wav(video_path):
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vn", "-ac", "1", "-ar", "22050", tmp],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return tmp


def audio_score(video_path):
    try:
        wav = extract_audio_wav(video_path)
        if not os.path.exists(wav) or os.path.getsize(wav) == 0:
            return 0.0
        y, sr = librosa.load(wav, mono=True, sr=22050)
        os.remove(wav)
        if len(y) == 0: return 0.0

        rms      = float(np.mean(librosa.feature.rms(y=y)))
        mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = float(np.mean(np.var(mfcc, axis=1)))
        S        = np.abs(librosa.stft(y))
        flux     = float(np.mean(np.diff(S, axis=1)**2))

        rms_n  = float(np.clip(rms * 10,         0, 1))
        var_n  = float(np.clip(mfcc_var / 500.0,  0, 1))
        flux_n = float(np.clip(flux / 0.5,        0, 1))

        return float(np.clip(0.50*rms_n + 0.30*var_n + 0.20*flux_n, 0.0, 1.0))
    except Exception:
        return 0.0