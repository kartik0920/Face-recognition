import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from scipy.spatial.distance import cosine

# Load the pre-trained FaceNet model (replace with your model path or TensorFlow Hub model)
model_url = "https://tfhub.dev/google/facenet/1"  # You can try a different URL or use a pre-trained model manually
face_model = tf.saved_model.load(model_url)

# Initialize MTCNN for face detection
detector = MTCNN()

def detect_faces(image):
    """Detect faces using MTCNN."""
    faces = detector.detect_faces(image)
    return faces

def extract_face_embeddings(image, faces):
    """Extract face embeddings for each detected face."""
    embeddings = []
    for face in faces:
        x, y, w, h = face['box']
        face_crop = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (160, 160))
        face_normalized = np.expand_dims(face_resized, axis=0) / 255.0
        embedding = face_model(face_normalized)
        embeddings.append(embedding.numpy().flatten())
    return embeddings

def compare_embeddings(embedding1, embedding2):
    """Compare two embeddings using cosine similarity."""
    return cosine(embedding1, embedding2)

# Load an image
image = cv2.imread('your_image.jpg')  # Replace with your image path
faces = detect_faces(image)

# Extract embeddings
embeddings = extract_face_embeddings(image, faces)

# Compare first two faces if available
if len(embeddings) > 1:
    similarity = compare_embeddings(embeddings[0], embeddings[1])
    print(f"Cosine Similarity: {similarity}")
    if similarity < 0.5:
        print("These faces belong to the same person.")
    else:
        print("These faces do not belong to the same person.")
else:
    print("No faces detected or only one face detected.")
