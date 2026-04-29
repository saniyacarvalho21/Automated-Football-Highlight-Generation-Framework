# MOTION SCORE — fast (every 10th frame), normalized [0,1]
import cv2, numpy as np

def motion_score(video):
    cap = cv2.VideoCapture(video)
    ret, prev = cap.read()
    if not ret:
        cap.release(); return 0.0
    prev   = cv2.cvtColor(cv2.resize(prev, (160, 90)), cv2.COLOR_BGR2GRAY)
    scores = []
    fc     = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        fc += 1
        if fc % 3 != 0: continue   # every 3rd frame only
        gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        scores.append(float(np.mean(cv2.absdiff(prev, gray))))
        prev = gray
    cap.release()
    if not scores: return 0.0
    return float(np.clip(np.mean(scores) / 255.0, 0.0, 1.0))