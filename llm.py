from google import genai
from dotenv import load_dotenv
load_dotenv()
client = genai.Client(api_key=os.getenv("Gemini_api_key"))
def call_gemini(prompt: str, model_name: str):
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    return response
