#Phase2:====================================
#Step1: Setup Pydantic Model
from pydantic import BaseModel
from typing import List

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages:List[str]
    allow_search:bool


#Step2: Setup AI agent from Front END request
from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent

ALLOWED_LLM_MODELS=["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "gpt-4o-mini"]

ALLOWED_LLM_PROVIDERS = ["openai","groq"]

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok", "message": "AI Agent is running"}

@app.post("/chat")
async def chat(request: RequestState):
    if request.model_name not in ALLOWED_LLM_MODELS:
        return {"error": "Invalid model name, Please select a valid model name"}
    if request.model_provider not in ALLOWED_LLM_PROVIDERS:
        return {"error": "Invalid model provider"}
    #Create AI agent and get respons efrom it:
    response = get_response_from_ai_agent(
        llm_id=request.model_name,
        query=request.messages[-1],
        allow_search=request.allow_search,
        system_prompt=request.system_prompt,
        provider=request.model_provider
    )
    return {"response": response}

#Step3: Run app & Explore swagger UI docs.
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="[IP_ADDRESS]", port=8000)
