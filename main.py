"""Main entrypoint for the app."""
import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_cohere_chain
from query_data import get_bedrock_chain
from schemas import ChatResponse
from langchain.embeddings import SentenceTransformerEmbeddings, CohereEmbeddings
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None


@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    if not Path("vectorstore.pkl").exists():
        raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    with open("vectorstore.pkl", "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)

    print(f"Vector store loaded successfully!")


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    #qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # qa_cohere_chain = get_cohere_chain(vectorstore)
    qa_chain = get_bedrock_chain(vectorstore)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            print("Waiting for incoming message!")
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            #result = await qa_chain.acall(
            #    {"question": question, "chat_history": chat_history}
            #)
            #chat_history.append((question, result["answer"]))
            result = qa_chain({"question": question, "chat_history": chat_history})
            print(f"Result from LLM: {result} ")

            metadata = None
            sources = []
            source_docs = None
            if "source_documents" in result:
                source_docs = result['source_documents']
                for doc in source_docs:
                    sources.append({"content": doc.page_content, "filename": doc.metadata['source']})

            '''if metadata:
                if "page" in metadata:
                    page = metadata["page"]
                    answer = f"{result['answer']} <br><br> Refer page {page} of {metadata['source']}"
                else:
                    answer = f"{result['answer']} <br><br> For more information, refer {metadata['source']}"
            else:
                answer = f"{result['answer']}"'''

            answer = f"{result['answer']}"

            stream_resp = ChatResponse(sender="bot", message=answer, type="stream")
            await websocket.send_json(stream_resp.dict())

            if source_docs:
                source_resp = ChatResponse(sender="bot", message=json.dumps(sources), type="sources")
                await websocket.send_json(source_resp.dict())

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())

            chat_history.append((question, result['answer']))
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

    uvicorn.run(app, host="0.0.0.0", port=3000)
