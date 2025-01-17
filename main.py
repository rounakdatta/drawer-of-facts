"""Main entrypoint for the app."""
import logging
import pickle
from pathlib import Path
from typing import Optional

import fastapi
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse, EmbeddingRequest

from vector_db import get_qdrant_impl, get_qdrant_client
from config import get_qdrant_config
from models import Information, MetaInformation
from ingest import ingest_docs
from datetime import datetime
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client.http import models as rest

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None
qdrant_config = get_qdrant_config()

VECTOR_SIZE = 1536
distance_func = "COSINE"

@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    global vectorstore
    vectorstore = get_qdrant_impl()
    
    qdrant_client = get_qdrant_client()
    try:
        # we check if the collection already exists
        existing_collection = qdrant_client.get_collection(qdrant_config["collection"])
    except:
        # if the collection doesn't exist, then we try create it
        qdrant_client.create_collection(
            collection_name=qdrant_config["collection"],
            vectors_config=rest.VectorParams(
                size=VECTOR_SIZE,
                distance=rest.Distance[distance_func],
            )
        )


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ingest")
async def ingest_data(request: EmbeddingRequest):
    ingest_docs(request.data)
    return fastapi.Response(content="OK", status_code=200)


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )

            source_answer_message = f" ::::: Source: {result['source_documents'][0].page_content}"
            end_resp = ChatResponse(sender="bot", message=source_answer_message, type="stream")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
