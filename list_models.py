# list_models.py

import google.generativeai as genai

genai.configure(api_key="AIzaSyAItZBq7-0L6LSJBFk3c04a26YEvs4NCNY")

for m in genai.list_models():
    print(f"{m.name} - {m.supported_generation_methods}")
