# Agentic Hierarchical RAG

A production-style Retrieval-Augmented Generation (RAG) system that combines **hierarchical document indexing**, **agentic reasoning**, and **multi-step retrieval** using LangGraph.

---

## 🚀 Overview

This project implements an advanced RAG pipeline that goes beyond standard vector search by:

* Building a **hierarchical tree index** over documents
* Performing **multi-level semantic retrieval**
* Using an **agent (LangGraph)** to dynamically decide:

  * when to retrieve
  * when to rewrite queries
  * when to generate answers
* Supporting **conversational memory** for multi-turn interactions

---

## 🧠 Architecture

```
PDFs
 ↓
Hierarchical Tree Index (semantic clustering)
 ↓
Hybrid Retriever (tree traversal)
 ↓
LangGraph Agent
 ├── History-aware query rewriting
 ├── Retrieval tool
 ├── Query refinement
 └── Answer generation
 ↓
Final Answer (grounded in context)
```

---

## 🔑 Features

* 📄 PDF ingestion and preprocessing
* 🌲 Hierarchical semantic chunking (tree-based indexing)
* 🔍 Context-aware retrieval with dynamic thresholds
* 🤖 Agentic RAG using LangGraph
* 💬 Conversational memory (chat history aware)
* 🧩 Modular design (retriever, generator, agent)

---

## 🛠️ Tech Stack

* Python
* LangChain
* LangGraph
* Sentence Transformers
* Scikit-learn (clustering)
* NumPy
* PyPDF
* NLTK

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

Download tokenizer:

```python
import nltk
nltk.download('punkt')
```

---

## ▶️ Usage

### 1. Build Index

```bash
python chunking.py
```

### 2. Run Agent

```python
from agent import chat

response = chat("Explain probability")
print(response)
```

---

## 💡 Example

```
User: What is probability?
Assistant: ...

User: What are its types?
Assistant: (uses conversation history to refine query)
```

---

## 📌 Future Improvements

* FAISS integration for faster retrieval
* RAG evaluation with RAGAS
* FastAPI deployment
* Frontend interface (Streamlit / React)
* Self-correcting agent loop

---

## 📖 Key Concepts

* Retrieval-Augmented Generation (RAG)
* Agentic AI
* Hierarchical document indexing
* Multi-step reasoning systems

---

## 👨‍💻 Author

Built as part of an advanced exploration into RAG systems and agentic workflows.

---
