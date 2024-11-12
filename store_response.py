import uuid
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from datetime import datetime, timezone
import dotenv
import os
import prompts

CHROMA_PATH = "store_path"
# hhh
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
llm = ChatOpenAI(model='gpt-4o-mini')

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

mongoclient = MongoClient(MONGO_URI)
db = mongoclient.chatbot

def store_text(text):
    website_id = str(uuid.uuid4())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(text)

    Chroma.from_documents(text_chunks, embeddings, persist_directory=CHROMA_PATH, collection_name=website_id)
    return website_id

def store_chat_history(question, answer, userid, chatbotid):
    chat_data = {
        "userId": userid,
        "chatbotId": chatbotid,
        "data": {
            "user": question,
            "bot": answer
        },
        "createdAt": datetime.now(timezone.utc) 
    }
    db.chats.insert_one(chat_data)


def query_bot(query, id):
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings, collection_name=id)

        # embedding = embeddings.embed_query(query)
        # docs = db.similarity_search_by_vector(embedding, k=3)  

        #retriever = db.as_retriever(
        #    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1}
        #)
        #docs = retriever.invoke(query)

        # retriever = db.as_retriever()
        # docs = retriever.invoke(query)

        docs = db.max_marginal_relevance_search(query)

        page_contents = "\n\n".join([doc.page_content for doc in docs])

        response = llm.invoke(f'Document:\n"{page_contents}"\n'+prompts.qbotprompt_template+query)
        return response.content

    except KeyError as e:
        return f"Error: Collection '{id}' not found in Chroma DB. {e}"
    
def chat_history(userid, chatbotid):
    history_list = list(db.chats.aggregate([
            {"$match": {"userId": userid, "chatbotId": chatbotid}},
            {"$sort": {"createdAt": 1}},
            {"$project": {"_id": 0, "data": 1}}
        ]))
    if history_list:
        return history_list
    return None

def chat_activity(chatbotid):
    history_list = list(db.chats.aggregate([
        {"$match": {"chatbotId": chatbotid}},
        {"$sort": {"createdAt": 1}},
        {"$project": {"_id": 1, "data": 1, "userId": 1}},  
        {
            "$group": {
                "_id": "$userId",  
                "messages": {
                    "$push": {
                        "data": "$data",
                        "objectId": {"$toString": "$_id"}  
                    }
                }
            }
        },
        {"$project": {"_id": 0, "userId": "$_id", "messages": 1}}  
    ]))
    if history_list:
        return history_list
    return None
    
def proper_query(question, userid, chatbotid):
    history_list = chat_history(userid, chatbotid)
    if history_list:
        history_text = "\n".join([f"User: {entry['data']['user']}\nBot: {entry['data']['bot']}" for entry in history_list])
        response = llm.invoke(f"Chat History:\n{history_text}\nUser Query: {question}\n"+prompts.contextualize_new)
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

def store_lead_info(user_id, chatbot_id, lead_number, lead_name):
    leads_collection = db["leads"]
    lead_data = {
        "user_id": user_id,
        "chatbot_id": chatbot_id,
        "lead_number": lead_number,
        "lead_name": lead_name
    }
    result = leads_collection.insert_one(lead_data)
