from fastapi import FastAPI
from pydantic import BaseModel
from debate import debate_pipeline, convert_ai_message_to_json

app = FastAPI()

# Define request model
class DebateRequest(BaseModel):
    query: str

# Define response model (assuming the structure of your AI message)
class DebateResponse(BaseModel):
    response: dict  # Adjust the type based on your actual response structure

@app.post("/debate/")  # Changed to POST since we're sending data
async def debate_api(request: DebateRequest) -> DebateResponse:
    ai_message = debate_pipeline(request.query)
    response_json = convert_ai_message_to_json(ai_message)
    return DebateResponse(response=response_json)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)