from fastapi import FastAPI, UploadFile, File
from pymongo import MongoClient
from config import MONGO_URI, DB_NAME
from cloudinary_utils import upload_file
import shutil
import os

app = FastAPI()
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Upload to cloudinary
    uploaded = upload_file(temp_path)
    os.remove(temp_path)

    # Save metadata to MongoDB
    db.datasets.insert_one({
        "filename": file.filename,
        "cloudinary_url": uploaded['secure_url'],
        "raw_public_id": uploaded['public_id'],
    })

    return {"message": "uploaded", "url": uploaded['secure_url']}
