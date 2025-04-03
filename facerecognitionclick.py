import cv2
import os
from deepface import DeepFace

# Path to the dataset (Face Database)
DATASET_DIR = "dataset/"

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def capture_and_recognize():

    cap = cv2.VideoCapture(0)  # Open the camera

    if not cap.isOpened():
        print("[ERROR] Could not open camera!")
        return

    print("[INFO] Press 'c' to capture an image...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture image.")
            break


        cv2.imshow("Press 'c' to Capture Image", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Press 'c' to capture
            captured_img_path = "captured_face.jpg"
            cv2.imwrite(captured_img_path, frame)
            print("[INFO] Image captured and saved!")
            break
        elif key == ord('q'):  # Press 'q' to exit
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    # Detect and recognize face
    recognize_face(captured_img_path)


def recognize_face(img_path):
    try:
        print("[INFO] Analyzing image...")

        # Use DeepFace to find the person in the dataset
        result = DeepFace.find(img_path=img_path, db_path=DATASET_DIR, model_name="ArcFace", detector_backend="mtcnn")

        if len(result[0]) > 0:
            best_match = result[0]['identity'][0]
            name = os.path.basename(os.path.dirname(best_match))  # Extract person's name
        else:
            name = "Unknown"

    except Exception as e:
        print("[ERROR] Face recognition failed:", str(e))
        name = "Unknown"

    # Show the captured image with the result
    img = cv2.imread(img_path)
    cv2.putText(img, name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)

    cv2.imshow("Recognized Face", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run the capture and recognition function
capture_and_recognize()
