import cv2
import face_recognition

# Load and encode known face
known_images = ["didi.jpg","jay.jpg","kaka.jpg","kewal.jpg","prince.jpg","navinya.jpg","nav1.jpg","nav2.jpg","nav3.jpg","nav4.jpg","nav5.jpg","kartik1.jpg","kartik2.jpg","kartik3.jpg"]  # Replace with your image
known_encodings = [face_recognition.face_encodings(face_recognition.load_image_file(img))[0] for img in known_images]
known_names = ["Sakshi", "jay","kaka","kewal","prince","navinya","navinya","navinya","navinya","navinya","navinya","kartik","kartik","kartik"]

# Label for the known face
# known_names = ["Sakshi"]  # Replace with person's name

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index] # If matched, assign the person's name

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Label the face
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
