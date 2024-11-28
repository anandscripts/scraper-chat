import uuid
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from datetime import datetime, timezone
import dotenv
import os
import prompts

CHROMA_PATH = "store_chroma"
FAISS_PATH = "store_faiss"

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
llm = ChatOpenAI(model='gpt-4o-mini')

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

mongoclient = MongoClient(MONGO_URI)
db = mongoclient.chatbot

def store_text(text):
    website_id = str(uuid.uuid4())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(text)

    Chroma.from_documents(text_chunks, embeddings, persist_directory=CHROMA_PATH, collection_name=website_id)
    # db = FAISS.from_documents(text_chunks, embeddings)
    # db.save_local(folder_path=FAISS_PATH, index_name=website_id)
    return website_id


def store_chat_history(question, answer, userid, chatbotid):
    chat_data = {
        "userId": userid,
        "chatbotId": chatbotid,
        "user": question,
        "bot": answer,
        "createdAt": datetime.now(timezone.utc) 
    }
    db.chats.insert_one(chat_data)


def query_bot(history, contextualized_question, user_query, id):
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings, collection_name=id)
        # db = FAISS.load_local(FAISS_PATH, embeddings, id, allow_dangerous_deserialization=True)

        # Retriever 1
        # embedding = embeddings.embed_query(contextualized_question)
        # docs = db.similarity_search_by_vector(embedding, k=3)  

        # Retriever 2
        # retriever = db.as_retriever(
        #    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1}
        # )
        # docs = retriever.invoke(contextualized_question)

        # Retriever 3
        retriever = db.as_retriever()
        docs = retriever.invoke(contextualized_question)

        # Retriever 4
        # docs = db.max_marginal_relevance_search(contextualized_question)

        page_contents = "\n\n".join([doc.page_content for doc in docs])
        response = llm.invoke(f'History:\n"{history}"\n\nDocuments:\n"{page_contents}"\n'+prompts.qbotprompt_new+user_query)
        return response.content

    except KeyError as e:
        return f"Error: Collection '{id}' not found in Chroma DB. {e}"
    
def chat_history(userid, chatbotid):
    history_list = list(db.chats.aggregate([
            {"$match": {"userId": userid, "chatbotId": chatbotid}},
            {"$sort": {"createdAt": 1}},
            {"$project": {"_id": 0,"user":1,"bot":1}}
        ]))
    if history_list:
        return history_list
    return None
  
def chat_activity(chatbotid):
    history_list = list(db.chats.aggregate([
        {"$match": {"chatbotId": chatbotid}},
        {"$sort": {"createdAt": 1}},
        {"$project": {"_id": 1, "user":1,"bot":1, "userId": 1}},  
        {
            "$group": {
                "_id": "$userId",  
                "messages": {
                    "$push": {
                        "user": "$user",
                        "bot": "$bot",
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
        history_text = "\n".join([f"User: {entry['user']}\nBot: {entry['bot']}" for entry in history_list])
        response = llm.invoke(f"Chat History:\n{history_text}\nUser Query: {question}\n"+prompts.contextualize_new)
        contextualized_question = response.content
    else:
        history_text = "No Previous Conversation"
        contextualized_question = question
    result = query_bot(history_text, contextualized_question, question, chatbotid)
    store_chat_history(question, result, userid, chatbotid)
    return result

def chatbot_details(chatbotid):
    history = db.Chatbot.find_one({"chatbotId": chatbotid})
    return history

def notification(userid, chatbotid):
    chatbot_detail = chatbot_details(chatbotid)
    print(chatbot_detail)
    if chatbot_detail:
        history_list = chat_history(userid, chatbotid)
        print(history_list)
        if history_list:
            history_text = "\n".join([f"User: {entry['user']}\nBot: {entry['bot']}" for entry in history_list[-5:]])
            print(history_text)
            response = llm.invoke(f"Chat History:\n{history_text}\n\nBased on this chat history, provide a helpful message to encourage the user to continue chatting. Respond only with the message.")
            print(response.content)
            return {"data": history_list, "response": response.content, "details": chatbot_detail}
        return {"data": None, "details": chatbot_detail}
    return {"data": None}

def store_lead_info(userid, chatbotid, name=None, number=None, purpose=None, requirement=None):
    collection = db['user_requests'] 
    try:
        lead_info = {}
        if userid: lead_info["userid"] = userid
        if chatbotid: lead_info["chatbotid"] = chatbotid
        if name: lead_info["name"] = name
        if number: lead_info["number"] = number
        if purpose: lead_info["purpose"] = purpose
        if requirement: lead_info["requirements"] = requirement

        result = collection.insert_one(lead_info)
        return str(result.inserted_id)
    except Exception as e:
        return f"An error occured: {str(e)}"
