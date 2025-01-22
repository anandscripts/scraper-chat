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
from bson import ObjectId
import openai
import json
# import app
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

def store_test(text):
    website_id = "nvnobvneri"
    collection_path = os.path.join(CHROMA_PATH, website_id)
    if os.path.exists(collection_path):
        shutil.rmtree(collection_path)
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(text)

    Chroma.from_documents(text_chunks, embeddings, persist_directory=CHROMA_PATH, collection_name=website_id)
    
    return website_id

def delete_chat_history(userid=None, chatbotid=None):
    # Build the query filter
    query = {}
    if userid:
        query['userId'] = userid
    if chatbotid:
        query['chatbotId'] = chatbotid

    # Delete matching documents
    result = db.chats.delete_many(query)
    
    # Print the result
    print(f"{result.deleted_count} chat(s) deleted.")

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


def query_bot(history, contextualized_question, user_query, chatbotid):
    try:
        db = mongoclient.chatbot
        prompts_collection = db.prompts  # Replace with your collection name

        # Fetch the single prompt from MongoDB
        prompt_data = prompts_collection.find_one({})
        if not prompt_data or "prompt" not in prompt_data:
            return "Error: Prompt not found in MongoDB."

        # Retrieve the prompt
        bot_prompt = prompt_data["prompt"]
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings, collection_name=chatbotid)
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

        response = llm.invoke(f'History:\n"{history}"\n\nDocuments:\n"{page_contents}"\n'+ bot_prompt +user_query)
        return response.content

    except KeyError as e:
        return f"Error: Collection '{id}' not found in Chroma DB. {e}"
    

def get_prompt():
    prompts_collection=db.prompts
    prompt_data = prompts_collection.find_one({})
    return {"prompt": prompt_data["prompt"]}

def update_prompt(prompt):
    """Update the prompt in MongoDB."""
    prompts_collection=db.prompts
    result = prompts_collection.update_one({}, {"$set": {"prompt": prompt}}, upsert=True)
    if result.modified_count > 0 or result.upserted_id:
        return {"message": "Prompt updated successfully."}

def chat_history(userid, chatbotid):
    history_list = list(db.chats.aggregate([
            {"$match": {"userId": userid, "chatbotId": chatbotid}},
            {"$sort": {"createdAt": 1}},
            {"$project": {"_id": 0, "data": 1}}
        ]))
    if history_list:
        return history_list
    return None

def getpage(id):
    result1=db.Chatbot
    result=list(result1.find({"chatbotId":id}, {"_id": 0,'userid':0}))
    print(result)
    if not result:
            return "Error: chatbotid not found in MongoDB."
    return result


def execute_function(function_name, parameters,userid,chatbotid):
    """Dispatch and execute the appropriate function."""
    if function_name == "store_lead_info":
         return store_lead_info(
            name=parameters.get("name", ""),
            number=parameters.get("number", ""),
            purpose=parameters.get("purpose", ""),
            requirement=parameters.get("requirement", ""),
            hist=parameters.get("hist", ""),
            uinput=parameters.get("uinput", ""),
            userid=userid,
            chatbotid=chatbotid
        )
    else:
        return {"error": "Unknown function called"}

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
    
# def proper_query(question, userid, chatbotid):
#     history_list = chat_history(userid, chatbotid)
#     if history_list:
#         history_text = "\n".join([f"User: {entry['data']['user']}\nBot: {entry['data']['bot']}" for entry in history_list])
#         response = llm.invoke(f"Chat History:\n{history_text}\nUser Query: {question}\n"+prompts.contextualize_new)
#         contextualized_question = response.content
#     else:
#         history_text = "No Previous Conversation"
#         contextualized_question = question
#     result = query_bot(history_text, contextualized_question, question, chatbotid)
#     store_chat_history(question, result, userid, chatbotid)
#     return result

def proper_query(user_input,userid, chatbotid):
    history_list = chat_history(userid, chatbotid)
    if not history_list:
        history_list="No history available"
    tools = [
    {
        "type": "function",
        "function": {
                    "name": "store_lead_info",
                    "description": "Capture lead details (name, phone, purpose, requirements) and store them in the Mongo database.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Full name of the lead."},
                            "phone": {"type": "string", "description": "Phone number of the lead, including country code."},
                            "purpose": {"type": "string", "description": "The purpose of the lead"},
                            "requirement": {"type": "string", "description": "Requirement of the lead"},
                            "hist": {"type": "string", "description": "Chat history"},
                            "uinput": {"type": "string", "description": "user input"},
                        },
                        "required": ["name", "phone", "purpose", "requirement","hist","uinput"]
                    }
                }
    }
    ]

    response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[ 
                {"role": "system", "content": "You are a helpful assistant and you have two jobs. 1)Given a chat history and the latest user question which might reference context in the chat history, formulate a contextualized standalone prompt which can be understood without the chat history, for RAG purposes.(Remove referential ambiguities by adding the reference from history) If it is in other language, translate the contextual query in English. Do NOT answer the user prompt, just reformulate it if needed and otherwise return it as is.2)Run the store_lead_info function when you have any of the parameters and still perform job "},#Given a chat history and the latest user question which might reference context in the chat history, formulate a contextualized standalone question which can be understood without the chat history. If it is in other language, translate the contextual query in English. Do NOT answer the user prompt, just reformulate it if needed and otherwise return it as is. Also if you have even one of the paramters for store_lead_info function call the function with other parameters as ''.even if you run the function answer the user prompt as well #You are a helpful assistant that can decide on executing functions based on user requests
                {"role": "user", "content": "Chat History:  " + str(history_list) + " User Prompt: " + user_input}
            ],
            tools=tools,
            tool_choice="auto"
            )

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            #print(tool_call)
            if tool_call.function.name=="store_lead_info":
                function_name = tool_call.function.name
                parameters = eval(tool_call.function.arguments)  # Safely parse arguments
                result = execute_function(function_name, parameters,userid,chatbotid)
                #print(f"Function executed: {function_name}, Result: {result}")
                break
    else:
        result=response.choices[0].message.content
    result=query_bot(str(history_list),result,user_input,chatbotid)
    store_chat_history(user_input, result, userid, chatbotid)
    return result
    #bot_response = response.choices[0].message.content
    #return bot_response

def notification(userid, chatbotid):
    history_list = chat_history(userid, chatbotid)
    if history_list:
        history_text = "\n".join([f"User: {entry['data']['user']}\nBot: {entry['data']['bot']}" for entry in history_list[-5:]])
        response = llm.invoke(f"Chat History:\n{history_text}\n\nBased on this chat history, provide a helpful message to encourage the user to continue chatting. Respond only with the message.")
        return {"data": history_list, "response": response.content}
    return {"data": None}

# def store_lead_info(userid, chatbotid, name=None, number=None, purpose=None, requirement=None):
#     collection = db['user_requests'] 
#     try:
#         lead_info = {}
#         if userid: lead_info["userid"] = userid
#         if chatbotid: lead_info["chatbotid"] = chatbotid
#         if name: lead_info["name"] = name
#         if number: lead_info["number"] = number
#         if purpose: lead_info["purpose"] = purpose
#         if requirement: lead_info["requirements"] = requirement

#         result = collection.insert_one(lead_info)
#         return str(result.inserted_id)
#     except Exception as e:
#         return f"An error occured: {str(e)}"


def store_lead_info(userid,chatbotid,name="", number="", purpose="", requirement="",hist="",uinput=""):
    collection = db['leads'] # Collection for user requests
    # print(hist)
    # print("Running\n")
    try:
        lead_info = {
            "name": name,
            "number": number,
            "purpose": purpose,
            "requirements": requirement,
            "userid":userid,
            "chatbotid":chatbotid
        }
        collection.insert_one(lead_info)
        result = collection.insert_one(lead_info)
        # return str(result.inserted_id)
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[ 
                {"role": "system", "content": "You are a helpful assistant .Given a chat history and the latest user prompt which might reference context in the chat history, formulate a contextualized standalone question which can be understood without the chat history.(Remove referential ambiguities by adding the reference from history) If it is in other language, translate the contextual query in English. Do NOT answer the user prompt, just reformulate it if needed and otherwise return it as is"},
                {"role": "user", "content": "Chat History:  "+ hist + "User Prompt:" + uinput}
            ])
        #print(response.choices[0].message.content+"\n")
        return (response.choices[0].message.content+"\n")
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
    