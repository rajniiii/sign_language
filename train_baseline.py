import os, numpy as np, glob, itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

INROOT = "data/keypoints"

def load_dataset(inroot=INROOT):
    X, y, labels = [], [], []
    # gather label names
    for label in sorted(os.listdir(inroot)):
        if not os.path.isdir(os.path.join(inroot, label)): 
            continue
        labels.append(label)
    if len(labels) < 2:
        raise RuntimeError("Need at least 2 labels in data/keypoints/<LABEL> to train a classifier.")

    # load .npy sequences per label
    for label in labels:
        seq_files = sorted(glob.glob(os.path.join(inroot, label, "*.npy")))
        for f in seq_files:
            seq = np.load(f)  # shape: [T, D] where D≈126
            # simple features: mean and std over time
            feat = np.concatenate([seq.mean(axis=0), seq.std(axis=0, ddof=1)])
            X.append(feat)
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y), labels

def main():
    X, y, labels = load_dataset()
    print("Feature dimension (train):", X.shape[1])  # should print 252

    print(f"Loaded {len(y)} samples across labels: {sorted(set(y))}")

    # stratified split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # scale features
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    # simple classifier
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)

    # evaluate
    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f"\nAccuracy: {acc*100:.2f}%\n")
    print("Per-class report:")
    print(classification_report(yte, ypred, labels=sorted(set(y)), zero_division=0))

    # confusion matrix
    cm = confusion_matrix(yte, ypred, labels=sorted(set(y)))
    print("Confusion matrix (rows=true, cols=pred):")
    print("Labels order:", sorted(set(y)))
    print(cm)
    
    # NEW: save artifacts for live use
    from joblib import dump
    os.makedirs("artifacts", exist_ok=True)
    labels_sorted = sorted(set(y))
    dump({"scaler": scaler, "clf": clf, "labels": labels_sorted}, "artifacts/baseline.joblib")
    print("Saved model -> artifacts/baseline.joblib")

if __name__ == "__main__":
    main()
