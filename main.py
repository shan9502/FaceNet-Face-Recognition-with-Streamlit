import streamlit as st
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from numpy.linalg import norm
import chromadb
import time
import asyncio
import uuid

st.set_page_config(
    page_title="Face Recognition",
    page_icon="icon/pngwing.com.png",
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
image_dir = 'face_images'

if "enable_monitoring" not in st.session_state:
    st.session_state.enable_monitoring = False

frame_placeholder = st.empty()

chroma_client = chromadb.HttpClient(host="localhost", port=8000)
chroma_collection_name = "facenet_embeddings"
collection = chroma_client.get_or_create_collection(name=chroma_collection_name,  metadata={"hnsw:space": "cosine"})
async def startMonitoring():
    # Start video capture
    # video_path = "My Projects/Ai based/ComputerVision/videos/videoplayback.mp4"
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with a video file path

    print("Starting video feed...")
    while st.session_state.enable_monitoring:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        # Convert frame to RGB (MTCNN expects RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # # Detect faces and extract bounding boxes
        boxes, _ = mtcnn.detect(rgb_frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Crop and extract embedding for the detected face
                face_tensor = mtcnn.extract(rgb_frame, [box], save_path=None).to(device)
                embedding = facenet(face_tensor).detach().cpu().numpy()[0]

                # Find the best match
                label = "Unknown"
                # best_similarity = 0.0
                similarity_result = collection.query(
                    query_embeddings=[embedding],
                    n_results=1,
                    include=['documents', 'distances']
                )
                distance_threshold = 0.3
                if similarity_result['distances'][0] and similarity_result['distances'][0][0] < distance_threshold:
                    label = similarity_result['documents'][0][0]
                else:
                    label = "Unknown"

                # Display the label
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 122, 98), 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Display the video feed with annotations
        frame_placeholder.image(frame_rgb)

        # cv2.imshow("FaceNet Recognition", frame)
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


@st.dialog("Upload Images") 
def imageUpload():
    uploaded_files = st.file_uploader("(Ensure faces of the same person accross images)", type=["jpg", "png", "jpeg"], accept_multiple_files=True,label_visibility="visible")
    name = st.text_input("Enter a name for the uploaded images", placeholder="John Doe")
    if st.button("Submit"):
        if len(uploaded_files)==0:
            st.error("Please upload an image", icon=None)
        elif not name:
            st.error("Please enter a name", icon=None)
        else:
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)

            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                add_face_img = add_known_face(name, bytes_data)
                if add_face_img:
                    st.toast(f"Image {uploaded_file.name} uploaded successfully", icon="✅")
                else:
                    st.toast(f"Image {uploaded_file.name} could not be uploaded", icon="❌")
            # st.success("Images uploaded successfully")
            time.sleep(.5)
            st.rerun()

with st.sidebar:
    st.write("Upload Images of persons to be recognized")
    if st.button("Upload Image"):
        imageUpload()

if st.button("Start Monitoring"):
        st.session_state.enable_monitoring = True
        asyncio.run(startMonitoring())

if st.button("Stop Monitoring"):
    st.session_state.enable_monitoring = False
    print("Stopping monitoring")


def add_known_face(name, image_bytes_data):
    try:
        name = str(name)
        nparr = np.frombuffer(image_bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        box, _ = mtcnn.detect(img_rgb)
        if box is not None:
            face_tensor = mtcnn.extract(img_rgb, [box[0]], save_path=None).to(device)
            embedding = facenet(face_tensor).detach().cpu().numpy()[0]
            collection.add(
                documents = name,
                embeddings = embedding,
                ids = str(uuid.uuid4())
                )
            # known_faces[name] = embedding
            print(f"Added {name} to the database.")
            return True
        return False
    except Exception as e:
        return False
