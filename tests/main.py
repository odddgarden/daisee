from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os

app = FastAPI()
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    try:
        file_location = os.path.join(UPLOAD_FOLDER, video.filename)
        with open(file_location, "wb+") as file_object:
            file_object.write(video.file.read())
        return JSONResponse(content={"message": "Video uploaded successfully", "path": file_location}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
