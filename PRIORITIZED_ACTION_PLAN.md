# 🚦 ModelShip Backend — Prioritized Action Plan

> **Important:** Complete each module fully before starting the next. Do **not** jump between modules. This ensures stability and avoids cross-module errors.

---

## 1️⃣ Finish Dataset Service (dataset_service)
- [x] Implement `/health` endpoint
- [x] Implement `/upload` endpoint (file upload, Cloudinary integration, MongoDB metadata)
- [ ] Add error handling/logging for upload failures
- [ ] (Optional) Implement `GET /datasets` and `GET /dataset/:id` for listing/fetching datasets
- [ ] Write tests for all endpoints
- [ ] Ensure `.env` config is robust and documented
- [ ] Final code review and refactor for clarity/modularity

**➡️ Only move to the next module when all above are complete and tested.**

---

## 2️⃣ Finish Labeling Service (labeling_service)
- [x] Implement `/health` endpoint
- [x] Implement `/label` endpoint (OpenAI integration, MongoDB save)
- [ ] Make `/label` endpoint async for consistency
- [ ] Add error handling/logging for OpenAI and DB failures
- [ ] Write tests for all endpoints
- [ ] Ensure `.env` config is robust and documented
- [ ] Final code review and refactor for clarity/modularity

**➡️ Only move to the next module when all above are complete and tested.**

---

## 3️⃣ Build Annotation Service (annotation_service)
- [ ] Implement `/health` endpoint
- [ ] Implement `/save-label` endpoint (manual label review, save to MongoDB)
- [ ] Implement `/labeled/:dataset_id` endpoint (fetch all reviewed labels)
- [ ] Implement `/export/:dataset_id` endpoint (export as CSV/JSON)
- [ ] Add error handling/logging
- [ ] Write tests for all endpoints
- [ ] Ensure `.env` config is robust and documented
- [ ] Final code review and refactor for clarity/modularity

**➡️ Only move to the next module when all above are complete and tested.**

---

## 4️⃣ (Optional) API Gateway (api_gateway)
- [ ] Design and implement proxy routing to all services
- [ ] Add middleware (auth, logging, rate limiting) as needed
- [ ] Write tests for all routes
- [ ] Ensure `.env` config is robust and documented
- [ ] Final code review and refactor for clarity/modularity

---

## 5️⃣ General/Shared
- [ ] Ensure all services use `.env` and document required variables
- [ ] Add or refactor shared utilities if needed
- [ ] Write integration tests across services
- [ ] Update documentation for deployment and usage

---

**Remember:**
- Do not start a new module until the current one is fully finished and tested.
- This approach prevents errors and ensures a stable, modular backend. 