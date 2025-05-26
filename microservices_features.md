
# 🔧 ModelShip Backend Microservices – Feature Overview

This document outlines all planned microservices for the ModelShip MVP and what each one is responsible for.

---

## 1️⃣ Dataset Service (`dataset_service`)

**Purpose:** Handles file uploads, storage, and dataset metadata.

**Core Features:**
- Accepts CSV or JSON file uploads
- Stores files in Cloudinary
- Saves file metadata in MongoDB (filename, user_id, URL)
- Future: Parses and saves individual data rows for processing

**Planned Endpoints:**
- `POST /upload`
- `GET /datasets` (optional)
- `GET /dataset/:id` (optional)

---

## 2️⃣ Labeling Service (`labeling_service`)

**Purpose:** Uses OpenAI to generate label predictions for uploaded data rows.

**Core Features:**
- Accepts raw text + list of label options
- Calls OpenAI API with generated prompt
- Returns predicted label
- Stores prediction result in MongoDB for history

**Endpoints:**
- `POST /label`
- `GET /health`

---

## 3️⃣ Annotation Service (`annotation_service`) – *Planned*

**Purpose:** Allows users to manually review/edit labels and export the final dataset.

**Core Features:**
- Accepts user-edited labels
- Stores final labels in MongoDB with `source = user`
- Provides export as CSV or JSON
- (Optional) versioning or audit trail support

**Planned Endpoints:**
- `POST /save-label`
- `GET /labeled/:dataset_id`
- `POST /export/:dataset_id`

---

## 4️⃣ (Optional) API Gateway (`api_gateway`) – *Optional for MVP*

**Purpose:** Acts as a single entrypoint for routing requests to backend services.

**Core Features:**
- Proxy requests from frontend to appropriate microservice
- Attach middleware (auth, logging, rate limiting)

**Planned Routes:**
- `/upload/* → dataset_service`
- `/label/* → labeling_service`
- `/review/* → annotation_service`

---

## 🌐 Shared Stack & Services

- MongoDB Atlas for all services
- Cloudinary for file storage (Cloudinary URLs saved to DB)
- Python FastAPI (async REST APIs)
- OpenAI GPT-3.5 for label generation (temp, swappable)

---

## 🧪 Testing & Dev Strategy

Each service must:
- Use `.env` for config
- Run independently via `uvicorn main:app --reload`
- Expose `/health` endpoint for status checks
- Log all significant events and errors

---

Let’s ship the most modular AI backend possible. 💼🚀
