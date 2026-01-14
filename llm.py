from google import genai

client = genai.Client()

def call_gemini(prompt: str, model_name: str):
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    return response
