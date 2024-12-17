import cv2
import os

def capture_gesture(label, save_path="data"):
    cap = cv2.VideoCapture(0)
    os.makedirs(f"{save_path}/{label}", exist_ok=True)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Capturing {label} - Press 's' to save, 'q' to quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capture Gesture", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            cv2.imwrite(f"{save_path}/{label}/{count}.jpg", frame)
            count += 1
        elif key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    capture_gesture(label="hello")
