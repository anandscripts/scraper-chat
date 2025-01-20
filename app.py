from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
from fastapi.responses import JSONResponse
import asyncio
import importlib
import prompts
from pydantic import BaseModel
from typing import List
from scrape_links import scrape_links, scrape_text
from store_response import store_text, proper_query, notification, chat_activity, delete_chat_history,get_prompt,update_prompt

app = FastAPI()

api2_router = APIRouter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

@api2_router.get("/links")
async def scrape(request: Request, url: str):
    visited_links = set()

    async def link_stream():
        async for link_message in scrape_links(url, visited_links):
            yield link_message
            await asyncio.sleep(0.1)  
    return EventSourceResponse(link_stream())

class LinksRequest(BaseModel):
    links: List[str]

@api2_router.post("/scrape")
async def scrape(request: LinksRequest):
    links = request.links  # This will get the links from the body
    text_data = scrape_text(links)  
    collection_id = store_text(text_data)
    return {"chatbotId": collection_id}

class TextRequest(BaseModel):
    textData: str

@api2_router.post("/texttrain")
async def scrape(request: TextRequest):
    text = request.textData  
    text_data = scrape_text(text)  
    collection_id = store_text(text_data)
    return {"chatbotId": collection_id}

class ResponseRequest(BaseModel):
    question: str
    userid: str
    chatbotid: str

@api2_router.post('/chatresponse')
async def response(request: ResponseRequest):
    result = proper_query(request.question, request.userid, request.chatbotid)
    return {"data": result}

@api2_router.get('/chathistory')
async def history(request: Request, userid: str, chatbotid: str):
    return notification(userid, chatbotid)

@api2_router.get('/chatactivity')
async def activity(request: Request, chatbotid: str):
    return chat_activity(chatbotid)

qbotpromptnew=prompts.qbotprompt_new

def prompt_change(prompt:str):
    global qbotpromptnew
    qbotpromptnew=prompt
    return {"message": f"Prompt updated successfully: {qbotpromptnew}"}

class PromptRequest(BaseModel):
    prompt:str

# Pydantic model for the prompt
class PromptUpdate(BaseModel):
    prompt: str

@api2_router.get("/prompt", response_model=dict)
def get_prompt1():
    return get_prompt()

@api2_router.put("/prompt")
def update_prompt1(prompt_update: PromptUpdate):
    return update_prompt(prompt_update.prompt)

@api2_router.delete('/reset')
async def delete_chat_historys(userid="dbfudovn",chatbotid="nvnobvneri"):
    return delete_chat_history(userid,chatbotid)

@api2_router.post('/testing')
async def testing(request: ResponseRequest,chatbotid="nvnobvneri"):
    result = proper_query(request.question, "dbfudovn",chatbotid)
    return {"data": result}

    
@api2_router.get("/")
async def hello():
    return JSONResponse(content={"response": "Hello!"})

app.include_router(api2_router, prefix='/chatlaps')

# uvicorn api2_router:api2_router --host 192.168.0.100 --port 5001 --reload
