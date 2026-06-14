import os
import cv2
import numpy as np
import mediapipe as mp
from glob import glob

IMGROOT = "data/raw"
OUTNPY  = "data/image_keypoints.npy"   # features [N,126]
OUTLBL  = "data/image_labels.npy"      # labels   [N]

mp_hands = mp.solutions.hands
D = 21 * 3 * 2  # 126

def kp_from_image(path, hands):
    img = cv2.imread(path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    vec = np.zeros(D, dtype=np.float32)
    if res.multi_hand_landmarks:
        off = 0
        for h in res.multi_hand_landmarks[:2]:
            tmp = []
            for lm in h.landmark:
                tmp.extend([lm.x, lm.y, lm.z])
            tmp = np.array(tmp[:63], dtype=np.float32)
            vec[off:off+63] = tmp
            off += 63
            if off >= D: break
    return vec

def main():
    X, y = [], []
    labels = [d for d in sorted(os.listdir(IMGROOT)) if os.path.isdir(os.path.join(IMGROOT, d))]
    if not labels:
        raise SystemExit("No folders in data/images")
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        for label in labels:
            for p in sorted(glob(os.path.join(IMGROOT, label, "*"))):
                if not p.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                v = kp_from_image(p, hands)
                if v is None: 
                    continue
                X.append(v)
                y.append(label)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    os.makedirs(os.path.dirname(OUTNPY), exist_ok=True)
    np.save(OUTNPY, X)
    np.save(OUTLBL, y)
    print("Saved", X.shape, "->", OUTNPY)
    print("Labels:", {l: int(np.sum(y==l)) for l in sorted(set(y))})

if __name__ == "__main__":
    main()
