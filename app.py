from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
from sentence_transformers import SentenceTransformer, util

# Load combined data
with open("combined_data.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Extract only text content
corpus = [doc["content"] for doc in documents]

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Create FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "TDS Virtual TA Semantic API is running."}

@app.post("/ask")
async def ask_question(request: Request):
    body = await request.json()
    question = body.get("question", "")
    if not question:
        return JSONResponse(status_code=400, content={"message": "No question provided."})

    # Encode the input question
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Find best match using cosine similarity
    scores = util.cos_sim(question_embedding, corpus_embeddings)[0]
    best_match_id = scores.argmax().item()
    best_score = scores[best_match_id].item()

    if best_score < 0.4:
        return JSONResponse(status_code=404, content={"message": "No relevant answer found."})

    return {
        "answer": corpus[best_match_id],
        "score": round(best_score, 3)
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=10000)
