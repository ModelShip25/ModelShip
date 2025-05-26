# 🧱 MongoDB Collections – ModelShip

## datasets
- filename: str
- cloudinary_url: str
- raw_public_id: str
- created_at: datetime (optional)

## predictions
- text: str
- predicted_label: str
- source: str ("ai")
- created_at: datetime (optional)

## annotations (planned)
- entry_id: ObjectId
- final_label: str
- user_id: str
- source: str ("user")
- timestamp: datetime