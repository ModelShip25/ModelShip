# 🚀 ModelShip API Complete Testing Guide with Postman

## 🔧 Initial Setup

**Base URL:** `http://localhost:8000` (or whatever port the server starts on)
**Required Tools:** Postman Desktop App
**Prerequisites:** Backend server must be running (see startup instructions below)

### 🚀 Starting the Server

**Option 1 - Smart Startup (Recommended):**
```bash
cd backend
python start_server.py
```
This automatically finds an available port and tells you what URL to use.

**Option 2 - Manual Startup:**
```bash
cd backend
python main.py
```
If port 8000 is busy, you'll need to kill the existing process or use a different port.

---

## 📋 Testing Sequence (Follow This Order)

### 1. ✅ Health Check (No Auth Required)
- **Method:** `GET`
- **URL:** `http://localhost:8000/health`
- **Headers:** None
- **Expected Response (200):**
  ```json
  {
    "status": "healthy",
    "service": "ModelShip API",
    "version": "1.0.0"
  }
  ```

### 2. ✅ API Root (No Auth Required)  
- **Method:** `GET`
- **URL:** `http://localhost:8000/`
- **Headers:** None
- **Expected Response (200):**
  ```json
  {
    "message": "ModelShip API is running!",
    "version": "1.0.0",
    "docs": "/docs",
    "status": "healthy"
  }
  ```

### 3. ✅ API Documentation (No Auth Required)
- **Method:** `GET`  
- **URL:** `http://localhost:8000/docs`
- **Headers:** None
- **Expected:** Interactive Swagger UI in browser

---

## 🔐 Authentication Flow

### Test 4: User Registration
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/auth/register`
- **Headers:** 
  ```
  Content-Type: application/json
  ```
- **Body (raw JSON):**
  ```json
  {
    "email": "testuser@modelship.com",
    "password": "SecurePass123!"
  }
  ```
- **Expected Response (201):**
  ```json
  {
    "message": "User registered successfully",
    "user_id": 1
  }
  ```

### Test 5: User Login
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/auth/login`
- **Headers:** 
  ```
  Content-Type: application/json
  ```
- **Body (raw JSON):**
  ```json
  {
    "email": "testuser@modelship.com", 
    "password": "SecurePass123!"
  }
  ```
- **Expected Response (200):**
  ```json
  {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer"
  }
  ```

**🔑 IMPORTANT:** Copy the `access_token` value - you'll need it for all authenticated requests!

### Test 6: Get Current User Info
- **Method:** `GET`
- **URL:** `http://localhost:8000/api/auth/me`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Expected Response (200):**
  ```json
  {
    "id": 1,
    "email": "testuser@modelship.com",
    "subscription_tier": "free",
    "credits_remaining": 100
  }
  ```

---

## 📁 File Management

### Test 7: Upload Single File
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/upload`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Body:** `form-data`
  - Key: `file`
  - Type: File
  - Value: Select any image (jpg, png, gif) or text file (txt, csv)
- **Expected Response (201):**
  ```json
  {
    "file_id": 1,
    "filename": "test-image.jpg",
    "file_size": 245760,
    "message": "File uploaded successfully"
  }
  ```

### Test 8: List User Files
- **Method:** `GET`
- **URL:** `http://localhost:8000/api/files`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Expected Response (200):**
  ```json
  {
    "files": [
      {
        "id": 1,
        "filename": "test-image.jpg",
        "file_size": 245760,
        "file_type": "jpg",
        "status": "uploaded",
        "created_at": "2025-01-19T21:30:00"
      }
    ]
  }
  ```

### Test 9: Delete File
- **Method:** `DELETE`
- **URL:** `http://localhost:8000/api/files/1`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Expected Response (200):**
  ```json
  {
    "message": "File deleted successfully"
  }
  ```

---

## 🤖 Image Classification

### Test 10: Single Image Classification
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/classify/image`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Body:** `form-data`
  - Key: `file`
  - Type: File
  - Value: Select an image file (jpg, png, gif)
- **Expected Response (200):**
  ```json
  {
    "predicted_label": "Egyptian cat",
    "confidence": 87.45,
    "processing_time": 1.23,
    "credits_remaining": 99
  }
  ```

### Test 11: Batch Image Classification
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/classify/batch?job_type=image`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Body:** `form-data`
  - Key: `files` (Note: You can add multiple files)
  - Type: File  
  - Value: Select 2-5 image files
- **Expected Response (200):**
  ```json
  {
    "job_id": 1,
    "status": "queued",
    "total_items": 3,
    "message": "Classification job started"
  }
  ```

### Test 12: Get All User Jobs
- **Method:** `GET`
- **URL:** `http://localhost:8000/api/classify/jobs`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Expected Response (200):**
  ```json
  {
    "jobs": [
      {
        "id": 1,
        "job_type": "image",
        "status": "completed",
        "total_items": 3,
        "completed_items": 3,
        "created_at": "2025-01-19T21:30:00",
        "completed_at": "2025-01-19T21:30:15"
      }
    ]
  }
  ```

### Test 13: Get Specific Job Status
- **Method:** `GET`
- **URL:** `http://localhost:8000/api/classify/jobs/1`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Expected Response (200):**
  ```json
  {
    "job_id": 1,
    "status": "completed",
    "job_type": "image",
    "total_items": 3,
    "completed_items": 3,
    "created_at": "2025-01-19T21:30:00",
    "completed_at": "2025-01-19T21:30:15",
    "user_id": 1
  }
  ```

### Test 14: Get Job Results
- **Method:** `GET`  
- **URL:** `http://localhost:8000/api/classify/results/1`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Expected Response (200):**
  ```json
  {
    "job_id": 1,
    "results": [
      {
        "id": 1,
        "filename": "image1.jpg",
        "predicted_label": "Egyptian cat", 
        "confidence": 87.45,
        "processing_time": 1.23,
        "status": "success",
        "created_at": "2025-01-19T21:30:05"
      },
      {
        "id": 2,
        "filename": "image2.jpg",
        "predicted_label": "Golden retriever",
        "confidence": 92.18,
        "processing_time": 1.45,
        "status": "success", 
        "created_at": "2025-01-19T21:30:08"
      }
    ],
    "statistics": {
      "total_results": 2,
      "successful": 2,
      "failed": 0,
      "average_confidence": 89.82
    }
  }
  ```

---

## 📊 Export & Download

### Test 15: Create CSV Export
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/export/create/1?export_format=csv&include_confidence=true`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Expected Response (200):**
  ```json
  {
    "export_filename": "job_1_results_20250119_213015.csv",
    "download_url": "/api/export/download/job_1_results_20250119_213015.csv",
    "file_size": 1024,
    "records_exported": 2
  }
  ```

### Test 16: Create JSON Export
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/export/create/1?export_format=json`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Expected Response (200):**
  ```json
  {
    "export_filename": "job_1_complete_20250119_213020.json",
    "download_url": "/api/export/download/job_1_complete_20250119_213020.json",
    "file_size": 2048,
    "records_exported": 2
  }
  ```

### Test 17: Create Summary Report
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/export/create/1?export_format=summary`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Expected Response (200):**
  ```json
  {
    "export_filename": "job_1_summary_20250119_213025.txt",
    "download_url": "/api/export/download/job_1_summary_20250119_213025.txt",
    "file_size": 1536,
    "records_exported": 2
  }
  ```

### Test 18: Download Export File
- **Method:** `GET`
- **URL:** `http://localhost:8000/api/export/download/job_1_results_20250119_213015.csv`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Expected:** File download with CSV content

### Test 19: Get Available Export Formats
- **Method:** `GET`
- **URL:** `http://localhost:8000/api/export/formats`
- **Headers:** None required
- **Expected Response (200):**
  ```json
  {
    "formats": [
      {
        "name": "csv",
        "description": "Comma-separated values with classification results",
        "file_extension": ".csv"
      },
      {
        "name": "json", 
        "description": "Complete job data with statistics in JSON format",
        "file_extension": ".json"
      },
      {
        "name": "summary",
        "description": "Human-readable summary report",
        "file_extension": ".txt"
      }
    ]
  }
  ```

---

## 🚫 Error Testing

### Test 20: Invalid Authentication
- **Method:** `GET`
- **URL:** `http://localhost:8000/api/auth/me`
- **Headers:** 
  ```
  Authorization: Bearer invalid_token_here
  ```
- **Expected Response (401):**
  ```json
  {
    "detail": "Could not validate credentials"
  }
  ```

### Test 21: Insufficient Credits
(First use up all credits by doing many classifications, then try again)
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/classify/image`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Body:** Upload an image
- **Expected Response (402):**
  ```json
  {
    "detail": "Insufficient credits"
  }
  ```

### Test 22: Invalid File Type
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/classify/image`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Body:** Upload a .pdf or .docx file
- **Expected Response (400):**
  ```json
  {
    "detail": "Invalid image type"
  }
  ```

### Test 23: Job Not Found
- **Method:** `GET`
- **URL:** `http://localhost:8000/api/classify/jobs/999`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Expected Response (404):**
  ```json
  {
    "detail": "Job not found"
  }
  ```

---

## 🔍 Advanced Testing Scenarios

### Test 24: Large Batch Processing
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/classify/batch?job_type=image`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Body:** Upload 10+ image files
- **Expected:** Job created successfully, then monitor status

### Test 25: Confidence Filtering
- **Method:** `POST`
- **URL:** `http://localhost:8000/api/export/create/1?export_format=csv&confidence_threshold=0.8`
- **Headers:** 
  ```
  Authorization: Bearer YOUR_ACCESS_TOKEN_HERE
  ```
- **Expected:** Only high-confidence results exported

---

## 📝 Quick Testing Checklist

- [ ] Health check works
- [ ] User can register
- [ ] User can login and get token
- [ ] File upload works
- [ ] Single image classification works
- [ ] Batch classification creates job
- [ ] Job status updates correctly
- [ ] Results are retrievable
- [ ] CSV export works
- [ ] JSON export works  
- [ ] File download works
- [ ] Error handling works for invalid auth
- [ ] Error handling works for invalid files

---

## 🛠️ Troubleshooting

**Database Errors (500):** Run `python recreate_db.py` in backend folder

**Import Errors:** Check all dependencies in `requirements.txt` are installed

**File Upload Issues:** Ensure `uploads/` directory exists in backend folder

**ML Model Issues:** First image classification may take longer as models download

**Token Expired:** Re-run the login test to get a fresh token

---

## 📊 Expected Performance Benchmarks

- **Health check:** < 50ms
- **User registration:** < 200ms  
- **File upload:** < 1s for files under 5MB
- **Single image classification:** 2-10s (first time), < 3s after
- **Batch job creation:** < 500ms
- **Export creation:** < 2s for 100 results

---

## 🎯 Success Criteria

✅ All endpoints return expected status codes  
✅ Authentication flow works end-to-end  
✅ File operations complete successfully  
✅ Image classification produces reasonable labels  
✅ Batch processing handles multiple files  
✅ Export formats generate correctly  
✅ Error cases return appropriate messages

---

## 🚀 Getting Started Quick Steps

1. **Start the backend server:** `cd backend && python main.py`
2. **Test health check first:** GET `http://localhost:8000/health`
3. **Register a user:** POST to `/api/auth/register`
4. **Login to get token:** POST to `/api/auth/login`
5. **Test single image classification:** POST to `/api/classify/image`
6. **Try batch processing:** POST to `/api/classify/batch`
7. **Check job status:** GET `/api/classify/jobs/{job_id}`
8. **Get results:** GET `/api/classify/results/{job_id}`
9. **Export results:** POST to `/api/export/create/{job_id}`

This comprehensive guide covers all implemented ModelShip API endpoints. Follow the sequence for best results! 