# 📜 API Contracts – ModelShip MVP

## POST /upload (Dataset Service)
Request: FormData (file)
Response:
{
  "message": "uploaded",
  "url": "https://res.cloudinary.com/.../dataset.csv"
}

---

## POST /label (Labeling Service)
Request:
{
  "text": "The phone lasts all day.",
  "options": ["Positive", "Neutral", "Negative"]
}

Response:
{
  "text": "The phone lasts all day.",
  "predicted_label": "Positive",
  "source": "ai"
}