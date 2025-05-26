
# 🧠 ModelShip Backend — Cursor AI Development Guide

---

## ⚙️ Project Overview

This is the backend-only implementation of **ModelShip**, an AI-powered data platform that enables users to:

- Upload datasets (CSV/JSON)
- Auto-label dataset rows using OpenAI
- Save and export labeled results
- (Upcoming) Allow manual editing of labels

Built with a microservice architecture using **FastAPI**, **MongoDB**, and **Cloudinary**.

---

## 🧩 MVP Features & Workflows

### ✅ Feature 1: Upload Dataset

**Workflow:**
1. User uploads CSV or JSON file via frontend
2. Backend sends file to Cloudinary
3. File metadata (filename, Cloudinary URL, user_id) is saved in MongoDB

**Route:**
- `POST /upload` (in `dataset_service`)

---

### ✅ Feature 2: Auto-Label Rows Using OpenAI

**Workflow:**
1. Frontend sends a single row of text and label options
2. Backend generates prompt and sends it to OpenAI
3. OpenAI response is saved to MongoDB
4. Returns predicted label to user

**Route:**
- `POST /label` (in `labeling_service`)

---

### 🔜 Feature 3: Save Manual Review of Labels (Upcoming)

**Workflow:**
1. User edits label manually
2. Frontend sends new label + row info to annotation_service
3. Save reviewed label in `annotations` collection

**Routes (planned):**
- `POST /save-label`
- `GET /labeled/:dataset_id`

---

### 🔜 Feature 4: Export Final Labeled Dataset (Upcoming)

**Workflow:**
1. User clicks “Export”
2. Backend retrieves all reviewed + labeled rows
3. Generates CSV or JSON file and returns for download

**Route (planned):**
- `POST /export/:dataset_id`

---

## 🧠 Cursor AI Instructions

> You are assisting with a microservice-based FastAPI backend for an AI data labeling MVP.  
> Help write clean, modular FastAPI code with MongoDB and Cloudinary support.  
> Services include: dataset upload, auto-labeling with OpenAI, manual annotation (upcoming).  
> Use async functions, `.env` configs, and RESTful design.  
> Focus on clarity, modularity, and API-first principles.

---

## 🛠 Tech Stack

- Python 3.11+
- FastAPI
- MongoDB Atlas
- Cloudinary
- OpenAI API (gpt-3.5-turbo)
- Uvicorn

---

## 📁 Directory Layout

```
modelship-backend/
├── dataset_service/
│   ├── main.py
│   ├── config.py
│   ├── cloudinary_utils.py
├── labeling_service/
│   ├── main.py
│   ├── config.py
│   ├── prompt_engine.py
├── annotation_service/      # (planned)
├── shared/
│   └── .env.example
```

---

## ✅ Setup Notes

- Each service runs standalone with Uvicorn
- Use `python-dotenv` to load env vars from `.env`
- MongoDB collections:
  - `datasets`
  - `predictions`
  - `annotations` (coming soon)

---

## 🧪 Testing

- Test services using Postman or Swagger docs (`/docs`)
- Validate health endpoints and main flows before wiring everything together

---

# Let's build a fast, modular backend that powers ModelShip's MVP launch. 🚢
