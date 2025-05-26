from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pymongo import MongoClient
from config import MONGO_URI, DB_NAME
from prompt_engine import label_text

app = FastAPI()
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

class LabelRequest(BaseModel):
    text: str
    options: List[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/label")
def auto_label(req: LabelRequest):
    try:
        label = label_text(req.text, req.options)
        result = {
            "text": req.text,
            "predicted_label": label,
            "source": "ai"
        }
        db.predictions.insert_one(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
