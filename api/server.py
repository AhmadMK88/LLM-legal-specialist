from fastapi import FastAPI, Request, HTTPException
from src.inference import generate_answer
from config.configs import API_KEY

app = FastAPI()

@app.post("/generate")
async def gen(req: Request):
    if req.headers.get("authorization") != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    data = await req.json()
    return {
        "response": generate_answer(data.get("prompt", ""))
    }
