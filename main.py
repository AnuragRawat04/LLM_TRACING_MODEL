from fastapi import FastAPI, HTTPException
from langfuse import observe,propagate_attributes
from langfuse_client import langfuse
from zoneinfo import ZoneInfo
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
import time
from datetime import datetime,timezone
from db import user_collection, logs_collection
from llm import call_gemini
from langfuse import get_client
langfuse_client=get_client()

app = FastAPI()
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

GoogleGenAIInstrumentor().instrument()
   
GEMINI_PRICING = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}

class InitRequest(BaseModel):
    name: str = Field(..., min_length=1)
    email: EmailStr


class PromptRequest(BaseModel):
    session_id:str
    email: EmailStr
    prompt: str = Field(..., min_length=1)
    model_name: str


class PromptResponse(BaseModel):
    response: str
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    estimated_cost: float

def calculate_cost(model_name: str, prompt_tokens: int, response_tokens: int) -> float:
    pricing = GEMINI_PRICING[model_name]
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (response_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


@app.post("/init_user")
async def init_user(data: InitRequest):
    existing = user_collection.find_one({"email": data.email})
    if existing:
        return {"message": "User already exists", "email": data.email}

    user_doc = {
        "name": data.name,
        "email": data.email,
        "created_at": datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Kolkata")),
    }
    user_collection.insert_one(user_doc)
    return {"message": "New user created", "email": data.email}

@app.post("/ask",response_model=PromptResponse)
@observe()
async def ask_llm(data: PromptRequest):
    if data.model_name not in GEMINI_PRICING:
        raise HTTPException(status_code=400, detail="Invalid model name")

    # Ensure user exists
    user = user_collection.find_one({"email": data.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    start_time = time.time()

    with propagate_attributes(
        user_id=data.email,
        session_id=data.session_id,
        tags=['gemini',data.model_name]
    ):
        
        try:
            response = call_gemini(data.prompt, data.model_name)
            response_text = response.text

            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = usage.prompt_token_count if usage else 0
            response_tokens = usage.candidates_token_count if usage else 0
            latency = round(time.time() - start_time, 3)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM Provider Error: {str(e)}")

    total_tokens = prompt_tokens + response_tokens
    estimated_cost = calculate_cost(data.model_name, prompt_tokens, response_tokens)
    logs_collection.insert_one(
        {
            "user_id": user["_id"],
            "email": user["email"],
            "model_name": data.model_name,
            "prompt": data.prompt,
            "response": response_text,
            "usage": {
                "prompt": prompt_tokens,
                "completion": response_tokens,
                "total": total_tokens,
            },
            "cost": estimated_cost,
            "latency": round(latency, 3),
            "created_at": datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Kolkata")),
        }
    )

    return PromptResponse(
        response=response_text,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        total_tokens=total_tokens,
        estimated_cost=estimated_cost
)
