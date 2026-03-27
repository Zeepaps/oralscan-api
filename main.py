from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os

app = FastAPI(title="OralScan AI API")

@app.get("/")
def home():
    return {"message": "OralScan AI API is running! (Light version - Model coming soon)"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded image temporarily just to confirm it works
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # For now, return dummy prediction (we'll replace with real model later)
        return {
            "predicted_class": 0,
            "class_name": "Class 0: Healthy (Demo Mode)",
            "confidence": 85.5,
            "message": "This is a demo response. Full model will be added later."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(file_location):
            os.remove(file_location)