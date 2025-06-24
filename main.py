from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.chatbot_logic import get_remedy_reply
import joblib
from collections import OrderedDict
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from services import report_interpreter
import json
import tempfile
import os
# Load trained model and encoders
model = joblib.load("model/knn_model.pkl")
mlb = joblib.load("model/symptom_binarizer.pkl")
le = joblib.load("model/label_encoder.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SymptomRequest(BaseModel):
    symptoms: list[str]


@app.post("/predict")
def predict(request: SymptomRequest):
    try:
        input_vector = mlb.transform([request.symptoms])
        if input_vector.sum() == 0:
            return {
                "error": (
                    "None of the provided symptoms match our database."
                )
            }

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
                    "Consult a doctor for confirmation.",
                    "Stay hydrated and rest.",
                    "Do not rely solely on online tools for diagnosis."
                ],
                "severity": "Varies"
            })

        # Remove duplicates and keep highest-ranked match
        results = list(
            OrderedDict((r["disease"], r) for r in results).values()
        )
        # Sort by confidence (highest first)
        results = sorted(
            results,
            key=lambda r: float(r["confidence"].strip('%')),
            reverse=True
        )

        # Optionally limit to top 5
        results = results[:5]

        return {"predictions": results}

    except Exception as e:
        return {"error": str(e)}


class RemedyRequest(BaseModel):
    user_input: str


@app.post("/chat")
async def chat_remedy(data: RemedyRequest):
    reply = get_remedy_reply(data.user_input)
    return {"reply": reply}


@app.post("/interpret-report")
async def interpret_report(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        if suffix.lower() in [".pdf"]:
            extracted_text = report_interpreter.extract_text_from_pdf(tmp_path)
        elif suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            extracted_text = report_interpreter.extract_text_from_image(tmp_path)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type."})

        if not extracted_text:
            return JSONResponse(status_code=400, content={"error": "No readable text found in file."})

        raw_response = report_interpreter.get_gemini_analysis(extracted_text, file.filename)

        try:
            cleaned = raw_response.strip().strip('```json').strip('```')
            parsed = json.loads(cleaned)
            return parsed
        except Exception:
            return {"error": "Gemini returned an unreadable format", "raw": raw_response}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})