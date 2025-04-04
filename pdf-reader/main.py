from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)

vector_store_initialized = False
initialization_error = None

def initialize_vector_store():
    global vector_store_initialized, initialization_error
    try:
        pdf_path = "example.pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
        print("Processing PDF...")
        text = extract_text_from_pdf(pdf_path)
        print(f"Extracted text (first 500 characters): {text[:500]}")  # Debug log

        if not text.strip():
            raise ValueError("PDF text extraction returned empty content")
        
        print("Creating text chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )
        text_chunks = text_splitter.split_text(text)
        print(f"Number of chunks: {len(text_chunks)}")  # Debug log
        print(f"Sample chunk: {text_chunks[0]}")       # Debug log

        if not text_chunks or not any(chunk.strip() for chunk in text_chunks):
            raise ValueError("No valid text chunks could be created")
        
        print("Creating vector store...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        dummy_embedding = embeddings.embed_query("test")
        print(f"Dummy embedding (first 5 values): {dummy_embedding[:5]}")  # Debug log

        if not dummy_embedding:
            raise ValueError("Failed to create embeddings")
        
        vector_store = FAISS.from_texts(
            texts=["Sample text"] if not text_chunks else text_chunks,
            embedding=embeddings
        )
        vector_store.save_local("faiss_index")
        
        vector_store_initialized = True
        print("Vector store initialized successfully")
    except Exception as e:
        initialization_error = str(e)
        print(f"Initialization failed: {e}")
        raise

def extract_text_from_pdf(pdf_path):
    """Robust PDF text extraction with multiple fallbacks"""
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    print(f"Error extracting page: {e}")
                    continue
        
        if not text.strip():
            text = "Could not extract text from PDF. Please ensure the file contains selectable text."
            
    except Exception as e:
        print(f"PDF processing error: {e}")
        text = "Failed to process PDF file. The file may be corrupted or password protected."
    
    return text

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        initialize_vector_store()
        yield
    except Exception as e:
        print(f"Application startup failed: {e}")
        yield  # Allow app to run in degraded mode

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
async def serve_frontend():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

@app.get("/health")
async def health_check():
    if initialization_error:
        return {
            "status": "error",
            "initialized": False,
            "error": initialization_error
        }
    return {
        "status": "ok", 
        "initialized": vector_store_initialized
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        if not vector_store_initialized:
            if initialization_error:
                return {
                    "status": "error",
                    "answer": f"Cannot process questions: {initialization_error}"
                }
            raise HTTPException(
                status_code=400,
                detail="PDF processing not completed yet"
            )
            
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print(f"Vector store loaded: {new_db}")  # Debug log

        docs = new_db.similarity_search(request.question)
        print(f"Similar documents found: {len(docs)}")  # Debug log

        chain = get_conversational_chain()
        response = chain.invoke({
            "input_documents": docs, 
            "question": request.question
        })
        
        print(f"Response: {response['output_text']}")  # Debug log
        return {
            "status": "success",
            "answer": response["output_text"]
        }
    except Exception as e:
        print(f"Error processing question: {e}")  # Debug log
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

def get_conversational_chain():
    prompt_template = """
    Answer the question using the context below. If unsure, say you don't know.
    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0.3
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

if __name__ == "__main__":
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
