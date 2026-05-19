from fastapi import FastAPI
from pydantic import BaseModel
import random
from fastapi.middleware.cors import CORSMiddleware

from sentence_transformers import SentenceTransformer

from function_timer import timeit
from _inferencer import Inferencer, InferencerResults

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


class InferenceRequest(BaseModel):
    input: str = ""

inferencer = Inferencer.load_inferencer("./outputs/inferencer.pkl")

@timeit
def encode_using_sentence_transformer(input: str):
    return inferencer.make_inference(input)



@app.post("/api/infer/all")
def infer_all(request: InferenceRequest) -> InferencerResults:
    return inferencer.make_inference(request.input)

