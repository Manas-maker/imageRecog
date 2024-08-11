import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

def create_embeddings(root_folder):
    # Initialize face analysis model
    detector = FaceAnalysis(name='buffalo_l')  # Use 'buffalo_l' or other models as needed
    detector.prepare(ctx_id=-1)

    embeddings = []  # List to store embeddings
    names = []  # List to store corresponding names (based on folder names)

    # Iterate through each person's folder
    for person_name in os.listdir(root_folder):
        person_folder = os.path.join(root_folder, person_name)

        if os.path.isdir(person_folder):
            person_embeddings = []

            # Iterate through each image in the person's folder
            for filename in os.listdir(person_folder):
                if filename.endswith(('.jpg', '.png', '.jpeg')):  # Check for image file types
                    img_path = os.path.join(person_folder, filename)
                    img = cv2.imread(img_path)

                    # Perform face detection
                    faces = detector.get(img)

                    if faces is not None and len(faces) > 0:
                        # Assuming we only want the first detected face
                        face_embedding = faces[0].normed_embedding
                        person_embeddings.append(face_embedding)

                        # Debugging: print the shape of the embedding
                        print(f"Processed {filename} for {person_name}: embedding shape {face_embedding.shape}")

            if person_embeddings:
                # Average the embeddings for the person
                avg_embedding = np.mean(person_embeddings, axis=0)
                embeddings.append(avg_embedding)
                names.append(person_name)

    # Convert embeddings to a NumPy array for consistency
    try:
        embeddings = np.array(embeddings)
    except ValueError as e:
        print(f"Error converting embeddings to array: {e}")
        print("Shapes of individual embeddings:", [emb.shape for emb in embeddings])
        return

    # Save embeddings and names to a .pkl file
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump((embeddings, names), f)

    print(f"Embeddings saved: {embeddings.shape} for {len(names)} people.")

if __name__ == '__main__':
    root_folder = 'static'  # Replace with your root folder path
    create_embeddings(root_folder)

