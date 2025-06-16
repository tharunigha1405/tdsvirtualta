# api/index.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import os

# Set cache path
os.environ["HF_HOME"] = "/tmp/huggingface"

app = FastAPI()

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load your data
with open("combined_data.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

corpus = [doc["content"] for doc in documents]

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

corpus_embeddings = torch.cat([get_embedding(text) for text in corpus])

@app.get("/")
def root():
    return {"message": "FastAPI app running on Vercel"}

@app.post("/ask")
async def ask_question(request: Request):
    body = await request.json()
    question = body.get("question", "")
    if not question:
        return JSONResponse(status_code=400, content={"message": "No question provided."})
    question_embedding = get_embedding(question)
    scores = F.cosine_similarity(question_embedding, corpus_embeddings)
    best_match_id = torch.argmax(scores).item()
    best_score = scores[best_match_id].item()
    if best_score < 0.4:
        return JSONResponse(status_code=404, content={"message": "No relevant answer found."})
    return {"answer": corpus[best_match_id], "score": round(best_score, 3)}

