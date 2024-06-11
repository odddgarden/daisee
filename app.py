import os
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import cv2
import streamlit as st
from PIL import Image as PILImage

def extract_frames(video_path, output_folder, interval_seconds=15):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(interval_seconds * fps)
    
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval_frames == 0:
            # Save the frame as an image file
            frame_filename = os.path.join(output_folder, f"image_{saved_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    st.write(f"Extracted {saved_count} frames and saved to {output_folder}")

def query_db(image_vdb, query, results=3):
    results = image_vdb.query(
        query_texts=[query],
        n_results=results,
        include=['uris', 'distances'])
    return results

def print_results(results):
    for idx, uri in enumerate(results['uris'][0]):
        st.write(f"ID: {results['ids'][0][idx]}")
        st.write(f"Distance: {results['distances'][0][idx]}")
        st.write(f"Path: {uri}")
        st.image(uri, width=300)
        st.write("\n")

# Streamlit app structure
st.title("Video Frame Extractor and Image Search")

# Ensure the 'uploaded_videos' directory exists
upload_folder = 'uploaded_videos'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Ensure the 'extracted_frames' directory exists
extracted_frames_folder = 'extracted_frames'
if not os.path.exists(extracted_frames_folder):
    os.makedirs(extracted_frames_folder)

# Video upload
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if video_file is not None:
    video_path = os.path.join(upload_folder, video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    st.success(f"Video uploaded to {video_path}")

    # Create a unique output folder for the extracted frames
    video_name = os.path.splitext(video_file.name)[0]
    unique_output_folder = os.path.join(extracted_frames_folder, video_name)

    # Extract frames
    interval_seconds = st.number_input("Frame extraction interval (seconds)", min_value=1, max_value=60, value=15)
    if st.button("Extract Frames"):
        extract_frames(video_path, unique_output_folder, interval_seconds)

        # Create image vector database
        st.write("Creating image vector database...")
        dataset_folder = unique_output_folder

        chroma_client = chromadb.PersistentClient(path="image_vdb")
        image_loader = ImageLoader()
        CLIP = OpenCLIPEmbeddingFunction()

        image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)

        ids = []
        uris = []

        for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
            if filename.endswith('.png'):
                file_path = os.path.join(dataset_folder, filename)
                unique_id = f"{video_name}_{i}"
                ids.append(unique_id)
                uris.append(file_path)

        image_vdb.add(ids=ids, uris=uris)
        st.write("Images added to the database.")

        # Store the image_vdb in the session state
        st.session_state['image_vdb'] = image_vdb

# Query the database
if 'image_vdb' in st.session_state:
    query = st.text_input("Enter a search query")
    if st.button("Search"):
        results = query_db(st.session_state['image_vdb'], query)
        print_results(results)
else:
    st.warning("Please upload a video and extract frames first.")
