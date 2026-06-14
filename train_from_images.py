import os
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression   # <-- import added
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

# ---------- optional but recommended: per-hand normalization ----------
def normalize_vec126(v: np.ndarray) -> np.ndarray:
    """Anchor each hand at its wrist and scale by wrist→middle-MCP distance."""
    v = v.copy().astype(np.float32)
    for h in range(2):
        base = h * 63
        hand = v[base:base+63]
        if not np.any(hand):
            continue
        pts = hand.reshape(21, 3)
        wrist_xy = pts[0, :2]
        ref_xy = pts[9, :2]   # middle finger MCP
        scale = float(np.linalg.norm(ref_xy - wrist_xy))
        if scale < 1e-6:
            scale = 1.0
        pts[:, :2] = (pts[:, :2] - wrist_xy) / scale
        v[base:base+63] = pts.reshape(-1)
    return v
# ---------------------------------------------------------------------

# Load features extracted from images
X = np.load("data/image_keypoints.npy")   # shape [N, 126]
y = np.load("data/image_labels.npy")      # shape [N]

# Normalize each sample (comment out if you haven't added this live)
X = np.apply_along_axis(normalize_vec126, 1, X)

print("Loaded", X.shape, "labels:", sorted(set(y)))
print("Counts:", Counter(y))

# Split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)

# -------- Balanced classifier (Option B) --------
clf = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",   # <-- handles class imbalance
    multi_class="auto",
    solver="lbfgs"
)
clf.fit(Xtr, ytr)
# -----------------------------------------------

# Evaluate
yp = clf.predict(Xte)
acc = accuracy_score(yte, yp)
print(f"\nAccuracy: {acc*100:.2f}%\n")
print("Per-class report:")
print(classification_report(yte, yp, labels=clf.classes_.tolist(), zero_division=0))
print("Confusion matrix (rows=true, cols=pred):")
print(clf.classes_.tolist())
print(confusion_matrix(yte, yp, labels=clf.classes_.tolist()))

# Save artifact using the model's class order
os.makedirs("artifacts", exist_ok=True)
dump({
    "scaler": scaler,
    "clf": clf,
    "labels": clf.classes_.tolist(),
    "mode": "frame"
}, "artifacts/image_frame_model.joblib")
print("Saved -> artifacts/image_frame_model.joblib")
