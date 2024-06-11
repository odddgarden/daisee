import os
import chromadb
import base64
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from IPython.display import Image, display, Markdown
import cv2

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
    print(f"Extracted {saved_count} frames and saved to {output_folder}")

video_path = 'D:/Programs/Cchat/videos/va.mp4'
output_folder = 'extracted_frames'
extract_frames(video_path, output_folder, interval_seconds=1)

dataset_folder='D:/Programs/Cchat/extracted_frames'

# Instantiate the ChromaDB CLient
chroma_client = chromadb.PersistentClient(path="D:/Programs/Cchat/image_vdb")
# Instantiate the ChromaDB Image Loader
image_loader = ImageLoader()
# Instantiate CLIP embeddings
CLIP = OpenCLIPEmbeddingFunction()

# Create the image vector database
image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function = CLIP, data_loader = image_loader)

# Initialize lists for ids and uris (uniform resource identifiers, which in this case is just the path to the image)
ids = []
uris = []

# Iterate over each file in the dataset folder
for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
    if filename.endswith('.png'):
        file_path = os.path.join(dataset_folder, filename)
        
        # Append id and uri to respective lists
        ids.append(str(i))
        uris.append(file_path)

# Assuming multimodal_db is already defined and available
image_vdb.add(
    ids=ids,
    uris=uris
)

print("Images added to the database.")

def query_db(query, results=3):
    results = image_vdb.query(
        query_texts=[query],
        n_results=results,
        include=['uris', 'distances'])
    return results

def print_results(results):
    for idx, uri in enumerate(results['uris'][0]):
        print(f"ID: {results['ids'][0][idx]}")
        print(f"Distance: {results['distances'][0][idx]}")
        print(f"Path: {uri}")
        display(Image(filename=uri, width=300))
        print("\n")

# Testing it out
query = 'looking up'
results = query_db(query)
print_results(results)

