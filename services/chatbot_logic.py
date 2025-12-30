import os
from dotenv import load_dotenv

# Try to import Gemini only if available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

# Load .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    chat = model.start_chat(history=[])
else:
    model = None
    chat = None


def get_remedy_reply(user_input: str) -> str:
    if not model or not chat:
        return "⚠️ Gemini AI not available in this deployment. Try again later."

    user_input = user_input.strip()
    if user_input.lower() in ["hi", "hello", "hey", "hii", "heyy", "yo"]:
        return "👋 Hello! I'm HomeDoc AI. How can I help you today?"
    elif user_input.lower() in ["bye", "goodbye", "see you", "take care"]:
        return "👋 Goodbye! Take care and stay healthy."

    prompt = (
        f"You are HomeDoc AI – a certified digital health assistant.\n"
        f"The user said: \"{user_input}\"\n\n"
        "🧠 Your job:\n"
        "- Detect the language of the user input.\n"
        "- Respond in the **same language** as the user's input.\n"
        "- Use a gentle, caring, and comforting tone — like a kind nurse.\n"
        "- Give 3 to 4 simple and practical **home remedies** using common household or natural ingredients (e.g., honey, turmeric, warm water, ginger, salt, tulsi, lemon, garlic, ajwain, ghee, black pepper, clove, cinnamon, cumin, mustard oil, etc.). Feel free to suggest other traditional or widely available ingredients from Indian kitchens, even if not listed here.\n"
        "- Present each remedy as a bullet point:\n"
        "    • Bold heading: short title of remedy\n"
        "    • 1-2 lines of how to prepare and use it\n"
        "- Avoid long explanations. Keep it short, warm, and clear.\n"
        "- If needed, give a brief safety warning (e.g., 'If it worsens, consult a doctor').\n"
        "- End with a warm question like: 'Would you like food suggestions?' or 'Need help with anything else?' in the same language.\n"
        "- Always reply in the **same language** as the user's input, whether it's Hindi, Marathi, Tamil, Telugu, Bengali, Gujarati, or any other Indian language.\n"
    )

    try:
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error generating response: {e}"
