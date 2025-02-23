from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "RAG_With_Memory"
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="DebateAI API", description="API for debating with contextual RAG", version="1.0")

# Load and process the PDF document once during startup
file_path = "dataset/debate_training_data_report2.pdf"
loader = PyPDFLoader(file_path)
doc = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(doc)
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=splits, embedding=gemini_embeddings)
retriever = vectorstore.as_retriever()

# Initialize the model
model = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    max_tokens=500,
)

# System prompt for DebateAI
system_prompt = (
    "You are DebateAI created for having constructive debates with the user and encourage user to give counter arguments to you. "
    "If you don't know the answer, say that you don't know. "
    "Be a little aggressive and oppose the views of the user constructively and provide counter arguments with proper proof. "
    "Use the following pieces of retrieved context to understand the user's question and provide a relevant answer. "
    "Use few sentences and keep the conversation concise.\n\n{context}"
    
)

# Chat prompt for question answering
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("system","When the user gives up the debate give them the evaluation of how they performed and how can they improve their debating skills."),
        ("system", "Remember we are having a debate on the topic '{topic}' and you speak {side} it.Be {mood}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Retriever prompt for history-aware retrieval
retriever_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history. "
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retriever_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
        
       
    ]
)

# Setup RAG chain with history
history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# In-memory store for chat histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Conversational RAG chain with history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Pydantic model for request body
class ChatRequest(BaseModel):
    input: str
    topic: str
    side: str
    mood: str
    session_id: str = "default"

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to DebateAI API. Use /chat to start debating!"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = conversational_rag_chain.invoke(
            {"input": request.input,"mood":request.mood,"topic":request.topic,"side":request.side},
            config={"configurable": {"session_id": request.session_id}}
        )
        return {"answer": response["answer"], "session_id": request.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    if session_id not in store:
        return {"session_id": session_id, "history": []}
    history = [
        {"role": "AI" if isinstance(msg, AIMessage) else "User", "content": msg.content}
        for msg in store[session_id].messages
    ]
    return {"session_id": session_id, "history": history}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)