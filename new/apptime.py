import cv2
import base64
import numpy as np
import pickle
import time
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
socketio = SocketIO(app)

# Load custom embeddings from .pkl file
with open('embeddings.pkl', 'rb') as f:
    embeddings, names = pickle.load(f)
    print(f"Loaded embeddings shape: {embeddings.shape}")

# Initialize face analysis model
detector = FaceAnalysis(name='buffalo_l')  # Use the appropriate model pack
detector.prepare(ctx_id=-1)

# Initialize timer
last_recognition_time = 0
recognition_interval = 5  # seconds

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('video_frame')
def handle_video_frame(data):
    global last_recognition_time

    # Check if 5 seconds have passed since the last recognition
    current_time = time.time()
    if current_time - last_recognition_time >= recognition_interval:
        last_recognition_time = current_time

        # Decode the received frame
        frame = base64.b64decode(data)
        np_array = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if img is None:
            print("Failed to decode the image")
            emit('response', {'image': None})
            return

        # Perform face detection
        faces = detector.get(img)  # This returns a list of detected faces

        if faces is not None and len(faces) > 0:
            for face in faces:
                bbox = face.bbox.astype(int)  # Get the bounding box
                face_embedding = face.normed_embedding  # Get the normalized embedding

                # Ensure face_embedding is in the correct shape
                if face_embedding.ndim == 1:  # 1D array
                    face_embedding = face_embedding.reshape(1, -1)  # Reshape to 2D

                # Compare with custom embeddings
                similarities = cosine_similarity(face_embedding, embeddings)  # Ensure embeddings are a 2D array
                max_similarity_index = np.argmax(similarities)
                confidence = similarities[0][max_similarity_index]

                # Draw bounding box and label
                label = f'Person: {names[max_similarity_index]}, Confidence: {confidence:.2f}'
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert the processed frame back to JPEG format for display
        _, jpeg = cv2.imencode('.jpg', img)
        if jpeg is not None:
            img_data = base64.b64encode(jpeg.tobytes()).decode('utf-8')
            # Emit the processed image back to the client
            emit('response', {'image': img_data})
        else:
            print("Failed to encode the image")
            emit('response', {'image': None})
    else:
        # Emit an empty response to avoid blocking the client-side loop
        emit('response', {'image': None})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
