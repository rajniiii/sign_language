import os, time, sys, cv2

def open_cam(index_list=(0,1,2)):
    for i in index_list:
        cap = cv2.VideoCapture(i)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(0.3)  # warm-up
        if cap.isOpened():
            return cap, i
        cap.release()
    raise RuntimeError("No camera available. Close Zoom/Meet and check permissions.")

LABEL = (sys.argv[1] if len(sys.argv) > 1 else "HELLO").strip().upper()
OUTROOT = os.path.join("data", "raw", LABEL)
os.makedirs(OUTROOT, exist_ok=True)

cap, used_idx = open_cam()
print(f"Recording label: {LABEL}  (camera index {used_idx})")
print("Controls: r = record ~3s clip | q = quit")

def record_clip(seconds=3, fps=15):
    frames = []
    start = time.time()
    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imshow("Record", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None
        frames.append(frame.copy())
        # pace to ~fps
        t = start + len(frames)/fps - time.time()
        if t > 0: time.sleep(t)
    return frames

sample_id = 0
while True:
    ok, frame = cap.read()
    if not ok: continue
    cv2.putText(frame, f"Label: {LABEL} | 'r' record (~3s) | 'q' quit",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Record", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == ord('r'):
        print("Recording...")
        clip = record_clip(seconds=3, fps=15)
        if clip is None: break
        tstamp = int(time.time())
        save_dir = os.path.join(OUTROOT, f"{LABEL}_{tstamp}_{sample_id:03d}")
        os.makedirs(save_dir, exist_ok=True)
        for i, f in enumerate(clip):
            cv2.imwrite(os.path.join(save_dir, f"f{i:03d}.jpg"), f)
        print(f"Saved {len(clip)} frames -> {save_dir}")
        sample_id += 1

cap.release()
cv2.destroyAllWindows()
print("Done.")
