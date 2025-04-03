import cv2
import os
from deepface import DeepFace
import shutil

# Path to the dataset (Face Database)
DATASET_DIR = "dataset/"  # Change to your dataset path

# Clear previous DeepFace cache
if os.path.exists(DATASET_DIR + "/representations_vgg_face.pkl"):
    os.remove(DATASET_DIR + "/representations_vgg_face.pkl")

print("[INFO] Database cache cleared. Rebuilding...")

# Load the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to recognize faces in real-time
def recognize_faces():
    print("[INFO] Starting webcam for real-time face recognition...")

    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            # Save detected face temporarily
            face_img_path = "detected_face.jpg"
            cv2.imwrite(face_img_path, face)

            # Recognize face using DeepFace
            try:
                result = DeepFace.find(img_path=face_img_path, db_path=DATASET_DIR, model_name="ArcFace",detector_backend="mtcnn",distance_metric="cosine")

                if len(result[0]) > 0:
                    best_match = result[0]['identity'][0]
                    name = os.path.basename(os.path.dirname(best_match))  # Extract person's name from the path
                else:
                    name = "Unknown"
            except:
                name = "Unknown"

            # Draw rectangle and label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("DeepFace Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run face recognition
recognize_faces()
