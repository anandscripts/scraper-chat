from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
from fastapi.responses import JSONResponse
import asyncio
from pydantic import BaseModel
from typing import List
from scrape_links import scrape_links, scrape_text
from store_response import store_text, proper_query, notification

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

@app.get("/links")
async def scrape(request: Request, url: str):
    visited_links = set()

    async def link_stream():
        async for link_message in scrape_links(url, visited_links):
            yield link_message
            await asyncio.sleep(0.1)  
    return EventSourceResponse(link_stream())

class LinksRequest(BaseModel):
    links: List[str]

@app.post("/scrape")
async def scrape(request: LinksRequest):
    links = request.links  # This will get the links from the body
    text_data = scrape_text(links)  
    collection_id = store_text(text_data)
    return {"chatbotId": collection_id}

class ResponseRequest(BaseModel):
    question: str
    userid: str
    chatbotid: str

@app.post('/chatresponse')
async def response(request: ResponseRequest):
    result = proper_query(request.question, request.userid, request.chatbotid)
    return {"data": result}

@app.get('/chathistory')
async def history(request: Request, userid: str, chatbotid: str):
    return notification(userid, chatbotid)

@app.get("/")
async def hello():
    return JSONResponse(content={"response": "Hello!"})

# uvicorn app:app --host 192.168.0.100 --port 5001 --reload
