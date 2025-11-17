from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import CharacterTextSplitter 
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.llms.ollama import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA 
# ... rest of the code

import os
import sys
# ... rest of the code

# ... (The rest of your main.py code is fine)

# --- The rest of your main.py code ---
# ...

# --- 2. Configuration ---
# Define file path and model settings
DATA_FILE = "speech.txt"
OLLAMA_MODEL = "tinyllama"
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Main Function for RAG Execution ---
def run_rag_pipeline():
    print("--- 1. Loading Document ---")
    try:
        # 1. Load the provided text file (speech.txt).
        loader = TextLoader(DATA_FILE)
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s).")
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please create it.")
        sys.exit(1)


    print("--- 2. Splitting Text into Chunks ---")
    # 2. Split the text into manageable chunks.
    # Simple Character Splitter is enough for a small text.
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Split document into {len(texts)} chunks.")


    print("--- 3. Creating Embeddings and Local Vector Store (Chroma) ---")
    # 4. Embeddings: HuggingFaceEmbeddings (Local, No API)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 3. Create Embeddings and store them in a local vector store (ChromaDB).
    # This will create a local folder named 'chroma_db'
    if os.path.exists(CHROMA_DB_DIR):
        print("Loading existing ChromaDB...")
        db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    else:
        print("Creating new ChromaDB and persisting...")
        # Create and persist the database
        db = Chroma.from_documents(
            documents=texts, 
            embedding=embeddings, 
            persist_directory=CHROMA_DB_DIR
        )
        db.persist()
        print("ChromaDB created and persisted.")


    print("--- 4. Initializing Ollama LLM and RAG Chain ---")
    # 5. LLM: Ollama with Mistral 7B (Local)
    llm = Ollama(model=OLLAMA_MODEL)
    
    # 5. Generate an answer by feeding the retrieved context and the question to an LLM.
    # Use the ChromaDB as the retriever source
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=db.as_retriever(),
        return_source_documents=True # Helpful for debugging/verification
    )


    print("\n=======================================================")
    print(f"System Ready. LLM: {OLLAMA_MODEL}, DB: ChromaDB")
    print("=======================================================\n")

    # --- 5. Interactive Q&A Loop ---
    print("Start querying the RAG system. Type 'exit' to quit.")
    
    while True:
        try:
            user_question = input("Your Question: ")
            if user_question.lower() in ["exit", "quit"]:
                break
            
            if not user_question.strip():
                continue
                
            # Retrieve relevant chunks and generate answer
            result = qa_chain.invoke({"query": user_question})
            
            # Print the final answer
            print("\n--- Answer ---")
            # Result is a dictionary, the answer is usually in the 'result' key
            print(result['result'].strip()) 
            print("--------------\n")

        except Exception as e:
            print(f"An error occurred: {e}")
            # Ensure the Chroma connection is closed gracefully
            # (though ChromaDB doesn't strictly require a close method in this usage)
            break

if __name__ == "__main__":
    run_rag_pipeline()