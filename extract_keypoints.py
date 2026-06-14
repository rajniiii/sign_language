import cv2, mediapipe as mp, os, numpy as np
from tqdm import tqdm

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

INROOT = "data/raw"
OUTROOT = "data/keypoints"
os.makedirs(OUTROOT, exist_ok=True)

def process_image(path, hands):
    img = cv2.imread(path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    D = 21 * 3 * 2  # two hands padded
    vec = np.zeros(D, dtype=np.float32)
    if result.multi_hand_landmarks:
        hands_list = result.multi_hand_landmarks[:2]
        off = 0
        for hand in hands_list:
            coords = []
            for lm in hand.landmark:
                coords.extend([lm.x, lm.y, lm.z])  # 63 values
            vec[off:off+63] = np.array(coords[:63], dtype=np.float32)
            off += 63
    return vec  # always 126-dim (zeros if a hand missing)

def process_folder(label_path):
    label = os.path.basename(label_path)
    out_label_dir = os.path.join(OUTROOT, label)
    os.makedirs(out_label_dir, exist_ok=True)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        for seq_folder in os.listdir(label_path):
            seq_path = os.path.join(label_path, seq_folder)
            if not os.path.isdir(seq_path):
                continue
            all_data = []
            for fname in sorted(os.listdir(seq_path)):
                if not fname.endswith(".jpg"):
                    continue
                fpath = os.path.join(seq_path, fname)
                kps = process_image(fpath, hands)
                if kps is None:
                    continue
                all_data.append(kps)  # kps is 126-dim always
            if all_data:
                all_data = np.array(all_data, dtype=np.float32)  # [T, 126]
                out_path = os.path.join(out_label_dir, seq_folder + ".npy")
                np.save(out_path, all_data)
                print(f"Saved {out_path} with shape {all_data.shape}")


def main():
    for label in os.listdir(INROOT):
        label_path = os.path.join(INROOT, label)
        if os.path.isdir(label_path):
            process_folder(label_path)

if __name__ == "__main__":
    main()
