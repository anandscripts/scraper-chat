from chromadb import chromadb
from chromadb.config import Settings
from google.oauth2 import service_account
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymongo import MongoClient
import uuid
from datetime import datetime, timezone

genai.configure(api_key="AIzaSyCHUIiSPaTTDC8-1WEf8YkQi8dYNUVgzjU")
model = genai.GenerativeModel("gemini-1.5-flash")

credentials = service_account.Credentials.from_service_account_file('sapient-flare.json')
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", credentials=credentials)
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# client = chromadb.Client(Settings())  # Using default settings for local storage
client = chromadb.PersistentClient(path="C:/Users/Gopi/Desktop/chatbot/new/")


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
    # Connect to MongoDB
    client = MongoClient("mongodb+srv://anand0123:Qwer1234@chatbotcluster.jvkun.mongodb.net/chatbot?retryWrites=true&w=majority&appName=ChatbotCluster")
    db = client.chatbot

    # Prepare the data to be stored in the database
    chat_data = {
        "userId": userid,
        "chatbotId": chatbotid,
        "data": {
            "user": question,
            "bot": answer
        },
        "createdAt": datetime.now(timezone.utc)  # Store the current timestamp
    }

    # Insert the chat data into the database
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
        response = model.generate_content(similar_text+"If at all i ask you how you got these information act smart and say you know everything. Prompt: "+prompt)
        return response.text
    except KeyError as e:
        return f"Error: Collection '{id}' not found in Chroma DB. {e}"

def properresponse(question, userid, chatbotid):
    client = MongoClient("mongodb+srv://anand0123:Qwer1234@chatbotcluster.jvkun.mongodb.net/chatbot?retryWrites=true&w=majority&appName=ChatbotCluster")
    db = client.chatbot
    chats = bool(db.chats.find_one({"chatbotId": chatbotid}))
    if chats:
        chat_history = list(db.chats.find({"userId": userid, "chatbotId": chatbotid}).sort("createdAt", 1))

        history_text = "\n".join([f"User: {entry['data']['user']}\nBot: {entry['data']['bot']}" for entry in chat_history])

        response = model.generate_content("Make a new contextualized query for me with the given chat history.Add every piece of context in the new query.No cross question,just the query"+f"Chat History:\n{history_text}\nQuery: {question}")
        response = response.text
    else:
        response = question

    result = query_bot(response, chatbotid)
    store_chat_history(question, result, userid, chatbotid)
    return result


# print(store_text(text.text))
# query_bot('What is the company name?', chatbotid)

question="Who is the technical support"
userid = 'user123'
chatbotid= '17387dfb-5307-4ea3-8e92-09f6518991aa'

# print(properresponse(question, userid, chatbotid))
