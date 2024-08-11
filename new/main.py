from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize FaceAnalysis with 'buffalo_l'
face_analysis = FaceAnalysis(name='buffalo_l')
face_analysis.prepare(ctx_id=-1)  # Use -1 for CPU
print("Face analysis model initialized successfully.")

# Load stored embeddings from .pkl files in the current directory
def load_embeddings(embedding_folder='.'):
    embeddings = {}
    for filename in os.listdir(embedding_folder):
        if filename.endswith('.pkl'):
            name = filename.split('.')[0]
            with open(os.path.join(embedding_folder, filename), 'rb') as f:
                embeddings[name] = pickle.load(f)  # Load embedding
    return embeddings

stored_embeddings = load_embeddings()  # Load embeddings from the current directory

def create_embedding(face):
    return face.embedding  # Get the embedding of the detected face

def compare_embeddings(new_embedding, stored_embeddings, threshold=0.5):
    for name, embedding in stored_embeddings.items():
        similarity = cosine_similarity([new_embedding], [embedding])
        if similarity >= threshold:  # Define a threshold for recognition
            return name
    return "Unknown"

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam
    while True:
        success, frame = cap.read()  # Read a frame from the webcam
        if not success:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        faces = face_analysis.get(img_rgb)  # Detect and recognize faces

        # Draw bounding boxes and labels for recognized faces
        for face in faces:
            new_embedding = create_embedding(face)
            name = compare_embeddings(new_embedding, stored_embeddings)

            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(frame, name, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Label

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()  # Convert to byte array

        # Yield the frame in a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Render the index template

@app.route('/video_feed')
def video_feed():
    # Return the response with the video feed
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting Flask-SocketIO app...")
    socketio.run(app, host='0.0.0.0', port=5200, debug=True)
    print("Flask-SocketIO app is running.")


