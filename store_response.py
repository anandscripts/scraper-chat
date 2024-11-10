from chromadb import chromadb
from chromadb.config import Settings
from google.oauth2 import service_account
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
import uuid
from datetime import datetime, timezone
from langchain_openai import ChatOpenAI
import os
import prompts
# import text

genai.configure(api_key="AIzaSyCHUIiSPaTTDC8-1WEf8YkQi8dYNUVgzjU")
model = genai.GenerativeModel("gemini-1.5-flash")

os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(model='gpt-4o-mini')

# credentials = service_account.Credentials.from_service_account_file("C:/Users/Gopi/Desktop/chatbot/scrape_api/sapient-flare-414821-67e257076a0d.json")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", credentials=credentials)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# client = chromadb.Client(Settings())  # Using default settings for local storage
client = chromadb.PersistentClient(path="C:/Users/Gopi/Desktop/chatbot/new/")

mongoclient = MongoClient("mongodb+srv://anand0123:Qwer1234@chatbotcluster.jvkun.mongodb.net/chatbot?retryWrites=true&w=majority&appName=ChatbotCluster")
db = mongoclient.chatbot


def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def get_embedding(text):
    vector = embeddings.embed_query(text)
    return vector

def generate_unique_id(index):
    return f"chunk_{id}_{index}"


def store_text(text):
    website_id = str(uuid.uuid4())
    # collection = client.get_collection(name=id)
    collection = client.create_collection(name= website_id)
    chunks = chunk_text(text, chunk_size=250)
    for idx, chunk in enumerate(chunks):
        vector = get_embedding(chunk)  
        unique_id = generate_unique_id(idx)  
        collection.add(documents=[chunk], embeddings=[vector], metadatas=[{"text": chunk}], ids=[unique_id])
        return website_id

def store_chat_history(question, answer, userid, chatbotid):
    chat_data = {
        "userId": userid,
        "chatbotId": chatbotid,
        "data": {
            "user": question,
            "bot": answer
        },
        "createdAt": datetime.now(timezone.utc)  # Store the current timestamp
    }
    db.chats.insert_one(chat_data)

def query_bot(prompt,id):
    prompt_vector = get_embedding(prompt)  
    try:
        collection = client.get_collection(name=id)
        # collection.peek()
        n_results=1
        results = collection.query(query_embeddings=[prompt_vector], n_results=n_results)
        similar_text =""
        #similar_text = results['documents'][0][0]
        for i in range(n_results):
            similar_text=similar_text+results['documents'][0][i]
            # print(similar_text)
        response = llm.invoke(similar_text+prompts.qbotprompt+prompt)
        return response.content
    except KeyError as e:
        return f"Error: Collection '{id}' not found in Chroma DB. {e}"
    
def chat_history(userid, chatbotid):
    chats = bool(db.chats.find_one({"chatbotId": chatbotid}))
    if chats:
        history_list = list(db.chats.aggregate([
            {"$match": {"userId": userid, "chatbotId": chatbotid}},
            {"$sort": {"createdAt": 1}},
            {"$project": {"_id": 0, "data": 1}}
        ]))
        return history_list
    return None
    
def properresponse(question, userid, chatbotid):
    history_list = chat_history(userid, chatbotid)
    if history_list:
        history_text = "\n".join([f"User: {entry['data']['user']}\nBot: {entry['data']['bot']}" for entry in history_list])
        response = llm.invoke(prompts.contextualize+f"Chat History:\n{history_text}\nQuery: {question}")
        response = response.content
    else:
        response = question
    result = query_bot(response, chatbotid)
    store_chat_history(question, result, userid, chatbotid)
    return result

def notification(userid, chatbotid):
    history_list = chat_history(userid, chatbotid)
    if history_list:
        history_text = "\n".join([f"User: {entry['data']['user']}\nBot: {entry['data']['bot']}" for entry in history_list[-5:]])
        response = llm.invoke(f"Chat History:\n{history_text}\n\nBased on this chat history, provide a helpful message to encourage the user to continue chatting. Respond only with the message.")
        return {"data": history_list, "response": response.content}
    return {"data": None}


# print(store_text(text.text))
# query_bot('What is the company name?', chatbotid)

question="Who is the technical support of the company"
userid = 'user123'
chatbotid= '17387dfb-5307-4ea3-8e92-09f6518991aa'

# print(properresponse(question, userid, chatbotid))
