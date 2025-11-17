AmbedkarGPT-Intern-Task: Local RAG Prototype
Kalpit Pvt Ltd, UK - AI Intern Hiring: Assignment 1
This repository contains a functional Retrieval-Augmented Generation (RAG) system built using LangChain to answer questions based solely on the provided text in speech.txt. The entire pipeline operates locally, adhering to the no-API, no-cost constraints.
⚙️ Technical Implementation

Component Framework/Model Used Constraint Adherence
Orchestration LangChain Required
Vector DB ChromaDB (Local) Required
Embeddings sentence-transformers/all-MiniLM-L6-v2 Required (HuggingFaceEmbeddings, Local)
LLM Ollama (tinyllama) Required

Note on LLM: Due to local resource limitations preventing the stable use of Mistral 7B, the LLM was switched to the more lightweight TinyLlama to ensure the successful demonstration of the RAG pipeline.

 How to Run the System
Prerequisites
Ollama service installed and running.
TinyLlama model pulled: ollama pull tinyllama
Python 3.8+ (with a virtual environment recommended).

Steps

1.Install Dependencies (using requirements.txt):
    pip install -r requirements.txt
2.Run the Script:
    python main.py

The script will automatically load the text, create/load the local vector database (chroma_db/), and launch an interactive prompt for Q&A.
📂 Repository Contents
main.py: The core RAG pipeline code.
requirements.txt: All necessary Python dependencies.
speech.txt: The source text from Dr. B.R. Ambedkar.
.gitignore: Excludes the virtual environment (venv/) and the local vector store (chroma_db/).