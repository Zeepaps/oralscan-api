from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = FastAPI(title="OralScan AI - MobileNetV2 API")

# Load the model when the API starts
try:
    model = load_model("model.keras")
    print("✅ MobileNetV2 model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.get("/")
def home():
    return {"message": "OralScan AI API is running! Use /predict to upload an image."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Read the uploaded image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize to what MobileNetV2 expects (224x224)
        img = img.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0   

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)

        class_names = ["Class 0: Healthy", "Class 1: Mild Condition", "Class 2: Severe Condition"]
        result = class_names[predicted_class]

        return {
            "predicted_class": predicted_class,
            "class_name": result,
            "confidence": round(confidence, 2),
            "message": "Prediction successful"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")