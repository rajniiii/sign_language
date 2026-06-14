import time
import cv2
import numpy as np
from collections import deque
from joblib import load
import mediapipe as mp
import pyttsx3

# ----- thresholds -----
REPEAT_EVERY = 1.2
PRESENCE_WINDOW = 20
MIN_PRESENT_FRAMES = 12
MIN_MOTION = 0.015
CONF_THRESHOLD = 0.93
STREAK_NEEDED = 12

# ----- load the SEQUENCE baseline (from recorded clips) -----
ART = load("artifacts/baseline.joblib")
SCALER, CLF = ART["scaler"], ART["clf"]
LABELS = ART.get("labels") or CLF.classes_.tolist()
print("Using sequence baseline. Scaler expects:", getattr(SCALER, "n_features_in_", "unknown"))
print("Labels:", LABELS)

mp_hands = mp.solutions.hands
engine = pyttsx3.init()
engine.setProperty("rate", 175)

D = 21 * 3 * 2  # 126 dims

def kp_vector_from_result(result):
    """Return fixed-length 126-dim vector (zeros if no hands)."""
    vec = np.zeros(D, dtype=np.float32)
    if not result.multi_hand_landmarks:
        return vec
    off = 0
    for h in result.multi_hand_landmarks[:2]:
        tmp = []
        for lm in h.landmark:
            tmp.extend([lm.x, lm.y, lm.z])  # 63 values
        tmp = np.array(tmp[:63], dtype=np.float32)
        vec[off:off + 63] = tmp
        off += 63
        if off >= D:
            break
    return vec

def features_from_buffer(buf_np: np.ndarray):
    """Sequence features: mean+std over [T,126] -> [1,252]."""
    mu = buf_np.mean(axis=0)         # [126]
    sd = buf_np.std(axis=0, ddof=1)  # [126]
    feat = np.concatenate([mu, sd])  # [252]
    return feat.reshape(1, -1)

def recent_presence_and_motion(buf: deque):
    """(present_ok, avg_motion_xy) over last PRESENCE_WINDOW frames."""
    if len(buf) < PRESENCE_WINDOW:
        return False, 0.0
    recent = np.stack(list(buf)[-PRESENCE_WINDOW:])  # [W,126]
    present_frames = np.count_nonzero(np.any(recent != 0.0, axis=1))
    present_ok = present_frames >= MIN_PRESENT_FRAMES

    wrist_xy = recent[:, 0:2]                       # first-hand wrist (x,y)
    steps = np.linalg.norm(np.diff(wrist_xy, axis=0), axis=1)  # [W-1]
    avg_motion = float(steps.mean()) if steps.size else 0.0
    return present_ok, avg_motion

def say_once(text):
    engine.stop()
    engine.say(text)
    engine.runAndWait()

# ----- video -----
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

BUF = deque(maxlen=60)
streak_label = None
streak_len = 0
cooldown_until = 0.0

with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    print("Live sign recognition. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)
        kp = kp_vector_from_result(res)
        BUF.append(kp)

        pred_label = "-"
        conf = 0.0
        present_ok, avg_motion = recent_presence_and_motion(BUF) if len(BUF) >= 20 else (False, 0.0)

        if present_ok and avg_motion >= MIN_MOTION:
            buf_np = np.stack(BUF)                     # [T,126]
            feats = features_from_buffer(buf_np)       # [1,252]
            feats = SCALER.transform(feats)
            probs = CLF.predict_proba(feats)[0]
            j = int(np.argmax(probs))
            pred_label, conf = LABELS[j], float(probs[j])
        else:
            streak_label = None
            streak_len = 0

        # streak smoothing
        if pred_label != "-":
            if pred_label == streak_label:
                streak_len += 1
            else:
                streak_label = pred_label
                streak_len = 1

        now = time.time()
        if (streak_len >= STREAK_NEEDED and
            conf >= CONF_THRESHOLD and
            present_ok and
            avg_motion >= MIN_MOTION and
            now >= cooldown_until):
            say_once(streak_label.capitalize())
            cooldown_until = now + REPEAT_EVERY

        # HUD
        hud1 = f"pred:{pred_label} conf:{conf:.2f} streak:{streak_len}"
        hud2 = f"present:{present_ok} motion:{avg_motion:.3f}"
        cv2.putText(frame, hud1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, hud2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
        cv2.imshow("Live Recognizer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
