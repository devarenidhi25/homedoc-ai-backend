from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.chatbot_logic import get_remedy_reply
from services import report_interpreter
from services.image_diagnosis import predict_image

import joblib
from collections import OrderedDict
from fastapi.responses import JSONResponse
import tempfile
import json
import os
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO)


# ---------- LOAD MODELS (ABSOLUTE PATHS) ----------
model = joblib.load(os.path.join(BASE_DIR, "model", "knn_model.pkl"))
mlb = joblib.load(os.path.join(BASE_DIR, "model", "symptom_binarizer.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "model", "label_encoder.pkl"))


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- HEALTH CHECK ----------
@app.get("/")
def root():
    return {"message": "API running"}


# ---------- SYMPTOM PREDICTOR ----------
class SymptomRequest(BaseModel):
    symptoms: list[str]


@app.post("/predict")
def predict(request: SymptomRequest):
    try:
        input_vector = mlb.transform([request.symptoms])

        if input_vector.sum() == 0:
            return {"error": "None of the provided symptoms match our database."}

        distances, indices = model.kneighbors(input_vector, n_neighbors=15)
        predicted_diseases = le.inverse_transform(model._y[indices[0]])

        results = []
        for disease, distance in zip(predicted_diseases, distances[0]):
            score = round(100 / (1 + distance), 2)
            results.append({
                "disease": disease,
                "confidence": f"{score}%",
                "description": f"Match based on symptoms: {disease}.",
                "recommendations": [
                    "Consult a doctor.",
                    "Stay hydrated and rest."
                ],
                "severity": "Varies"
            })

        results = list(OrderedDict((r["disease"], r) for r in results).values())
        results = sorted(results, key=lambda r: float(r["confidence"].strip('%')), reverse=True)
        return {"predictions": results[:5]}

    except Exception as e:
        logging.error(e)
        return {"error": str(e)}


# ---------- CHATBOT ----------
class RemedyRequest(BaseModel):
    user_input: str


@app.post("/chat")
async def chat_remedy(data: RemedyRequest):
    reply = get_remedy_reply(data.user_input)
    return {"reply": reply}


# ---------- REPORT INTERPRETER ----------
@app.post("/interpret-report")
async def interpret_report(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        if suffix.lower() == ".pdf":
            extracted_text = report_interpreter.extract_text_from_pdf(tmp_path)
        else:
            extracted_text = report_interpreter.extract_text_from_image(tmp_path)

        if not extracted_text:
            return JSONResponse(status_code=400, content={"error": "No readable text found"})

        raw = report_interpreter.get_gemini_analysis(extracted_text, file.filename)

        try:
            cleaned = raw.strip().strip("```json").strip("```")
            return json.loads(cleaned)
        except:
            return {"raw": raw, "error": "Gemini returned unstructured output"}

    except Exception as e:
        logging.error(e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------- IMAGE DIAGNOSIS ----------
@app.post("/predict-image")
async def predict_medical_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_bytes = await file.read()

    try:
        return predict_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
