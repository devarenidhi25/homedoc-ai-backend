import os
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import pytesseract

# Load environment variables
load_dotenv()

# Check if the GEMINI_API_KEY is set
API_KEY = os.getenv("GEMINI_API_KEY")

genai = None
model = None

if API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("models/gemini-1.5-flash")
    except Exception as e:
        print("⚠️ Failed to initialize Gemini:", e)
else:
    print("⚠️ GEMINI_API_KEY not set. Gemini features will be disabled.")


def extract_text_from_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text.strip()


def extract_text_from_image(file_path):
    img = Image.open(file_path)
    return pytesseract.image_to_string(img).strip()


def get_gemini_analysis(text, file_name):
    if not model:
        return {
            "error": "Gemini model not available. Make sure GEMINI_API_KEY is set and package is installed."
        }

    prompt = f'''
You are a helpful, certified digital medical assistant.
A user has uploaded a medical report named: {file_name}.

Extracted Text:
{text}

Please analyze the report and return a JSON object with the following structure:

1. reportType: (e.g., "Blood Test", "X-ray", "Ultrasound", etc.)
2. keyFindings: (A list of 3–5 brief, patient-friendly bullet points summarizing important medical findings)
3. detailedAnalysis:
    - summary: A short summary of what the report means in simple language for the patient.
    - recommendations: A list of 3–4 easy-to-follow next steps the patient can take. Include lifestyle advice or self-care tips if applicable.
    - riskFactors: A list of any mentioned or inferred medical risk factors (e.g., obesity, smoking, cancer history). Explain each in layman's terms.

**Guidelines:**
- Use simple, non-technical language.
- Highlight if a finding is benign, serious, or needs monitoring.
- Include examples if helpful (e.g., "fatty liver can improve with weight loss and reducing sugar/alcohol intake").
- If breast cancer is mentioned, briefly explain what BIRADS means.

Output JSON only. Do not include extra commentary or markdown. Keep it concise but helpful.
'''
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return {"error": f"Gemini generation failed: {e}"}
