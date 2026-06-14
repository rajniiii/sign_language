import cv2

def open_camera(cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {cam_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit the preview window.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed. Trying again...")
            continue
        cv2.imshow("Camera Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        open_camera(0)
    except RuntimeError as e:
        print(e)
        print("Retrying with camera index 1...")
        open_camera(1)
