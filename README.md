# 🧠 Generative AI RAG with LangChain

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG) pipeline** using **LangChain**, **text splitters**, **embeddings**, and **vector databases** like **FAISS** or **Chroma**.  
It allows you to **ingest documents, store them as embeddings, and query them with a local LLM (Ollama/Hugging Face)**.

---

## ✨ Features
- 📄 Load documents (PDF, text, web pages)  
- ✂️ Split large docs into smaller chunks  
- 🔢 Convert text into embeddings (vectors)  
- 📦 Store embeddings in FAISS or ChromaDB  
- 🔍 Retrieve top-k relevant chunks for a query  
- 💬 Ask questions and get LLM-generated answers with context  

---

## 🛠️ Tech Stack
- **LangChain** – framework for LLM apps  
- **Embeddings** – HuggingFace / Ollama  
- **Vector Store** – FAISS / Chroma  
- **Text Splitters** – RecursiveCharacterTextSplitter  
- **LLMs** – Ollama (Gemma, Llama2, Mistral) or Hugging Face  

---

## 📂 Project Structure
```

├── data/                 # Documents to ingest
├── notebooks/            # Jupyter notebooks for experimentation
├── main.py               # Main RAG pipeline
├── requirements.txt      # Dependencies
└── README.md             # Project documentation

````

---

## ⚡ Quick Start

### 1. Clone Repo & Setup
```bash
git clone https://github.com/your-username/genai-rag.git
cd genai-rag
python -m venv .venv
.venv\Scripts\activate   # On Windows
pip install -r requirements.txt
````

### 2. Install Dependencies

```bash
pip install langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu chromadb python-dotenv
```

If using **Ollama**, also install it from [ollama.ai](https://ollama.ai) and pull a model:

```bash
ollama pull gemma:2b
```

### 3. Add Environment Variables

Create a `.env` file:

```
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

---

## 🚀 Usage

### Run with FAISS + Hugging Face Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load document
loader = TextLoader("data/sample.txt")
docs = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in FAISS
db = FAISS.from_documents(chunks, embeddings)

# Query
query = "What is Artificial Intelligence?"
results = db.similarity_search(query, k=2)
for r in results:
    print(r.page_content)
```

### Run with Ollama Embeddings + Local LLM

```python
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

embeddings = OllamaEmbeddings(model="gemma:2b")
db = Chroma.from_documents(chunks, embeddings)

retriever = db.as_retriever()
llm = ChatOllama(model="gemma:2b")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
print(qa.run("Summarize the document in 3 lines."))
```

---

## 📌 Next Steps

* Add **Conversational Retrieval Chain** for chat history
* Deploy as a **Streamlit web app**
* Experiment with **different LLMs** (Gemma, Llama2, Mistral, etc.)
* Try **hybrid search** (BM25 + embeddings)

---

## 📜 License

This project is open-source and free to use for educational purposes.


