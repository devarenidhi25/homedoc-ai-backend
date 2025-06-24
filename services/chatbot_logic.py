import os
from dotenv import load_dotenv
import google.generativeai as genai

# ========== CONFIGURATION ==========

# Load environment variables from .env file
load_dotenv()

# Fetch API key securely
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment.")

# Configure Gemini client
genai.configure(api_key=API_KEY)

# Initialize the model and start a chat session
model = genai.GenerativeModel("models/gemini-1.5-flash")
chat = model.start_chat(history=[])


def get_remedy_reply(user_input: str) -> str:
    user_input = user_input.strip()
    # 💬 Handle greetings simply (no language detection needed here)
    if user_input.lower() in ["hi", "hello", "hey", "hii", "heyy", "yo"]:
        return "👋 Hello! I'm HomeDoc AI. How can I help you today? You can describe your symptoms or ask for a home remedy."
    # 💬 Handle goodbye messages
    elif user_input.lower() in ["bye", "goodbye", "see you", "take care"]:
        return "👋 Goodbye! Take care and stay healthy. If you need help later, just ask!"
    prompt = (
        f"You are HomeDoc AI – a certified digital health assistant.\n"
        f"The user said: \"{user_input}\"\n\n"
        "🧠 Your job:\n"
        "- Detect the language of the user input.\n"
        "- Respond in the **same language** as the user's input.\n"
        "- Use a gentle, caring, and comforting tone — like a kind nurse.\n"
        "- Give 3 to 4 practical and simple **home remedies** using common household ingredients (e.g., honey, turmeric, warm water, ginger, salt, etc.).\n"
        "- Present each remedy as a bullet point:\n"
        "    • Bold heading: short title of remedy\n"
        "    • 1-2 lines of how to prepare and use it\n"
        "- Avoid long explanations. Keep it short, warm, and clear.\n"
        "- If needed, give a brief safety warning (e.g., 'If it worsens, consult a doctor').\n"
        "- End with a warm question like: 'Would you like food suggestions?' or 'Need help with anything else?' in the same language.\n"
        "- If user writes in Hindi or Marathi, reply in that language only.\n"
    )

    try:
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error generating response: {e}"
