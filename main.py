import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = FastAPI()

model_path = "c:/Users/admin/Desktop/8980/DL/DL_test/covid_classifier_balanced.h5"
trained_model = load_model(model_path)

class_names = ['COVID19', 'NORMAL', 'PNEUMONIA']

@app.get('/')
def get():
    return {"message": "Hello! Welcome to the COVID Classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        img = Image.open(io.BytesIO(contents)).convert("L")  
        img = img.resize((256, 256))
        img_array = img_to_array(img) / 255.0  

        img_array = np.repeat(img_array, 3, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = trained_model.predict(img_array)
        class_idx = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return {
            "predicted_class": class_names[class_idx],
            "confidence": f"{confidence:.2f}"
        }

    except Exception as e:
        return {"error": str(e)}