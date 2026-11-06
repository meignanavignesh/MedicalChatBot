# 🏥 Medical AI Chatbot (RAG-Powered)

An intelligent, domain-specific Medical Chatbot powered by **Retrieval-Augmented Generation (RAG)**, combining **BioBERT** medical embeddings, **FAISS** vector search, and **Groq LLaMA 3.1** high-speed LLM inference.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA--3.1-orange.svg)](https://groq.com/)

---

## 🌟 Overview

The **Medical Chatbot** retrieves relevant context from a curated medical knowledge base (including the Gale Encyclopedia of Medicine and clinical datasets) to provide accurate, grounded answers to healthcare questions without hallucination.

### Key Architecture:
- **Frontend / UI**: Streamlit Chat Interface (`st.chat_message`, `st.chat_input`).
- **Embeddings**: `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` (domain-tuned on biomedical NLI).
- **Vector Store**: FAISS (Facebook AI Similarity Search) index pre-indexed for sub-second retrieval.
- **LLM**: Groq LLaMA 3.1 (`llama-3.1-8b-instant`) for fast response generation.
- **Orchestration**: LangChain `create_retrieval_chain` + `create_stuff_documents_chain`.

---

## 🚀 Quick Start (Local)

### 1. Clone the Repository
```bash
git clone https://github.com/meignanavignesh/MedicalChatBot.git
cd MedicalChatBot
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```
*(Get a free API key at [console.groq.com](https://console.groq.com/keys))*

### 5. Run the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Fork or push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**:
   - **Repository**: `meignanavignesh/MedicalChatBot`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Advanced settings** → **Secrets**, and enter:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
5. Click **Deploy!** Your app will be live at `https://<your-custom-name>.streamlit.app`.

---

## 📂 Repository Structure

```text
├── app.py                      # Production entry point
├── medibot.py                  # Streamlit application logic
├── connect_memory_with_llm.py  # CLI inference script
├── requirements.txt            # Production dependencies
├── .gitignore                  # Clean gitignore configuration
├── vectorstore/                # Pre-built FAISS vector databases
│   └── db_faiss_sequential/    # Active sequential BioBERT FAISS index
├── data/                       # Reference datasets & clinical literature
└── usingDataset/               # MedQuAD dataset pipelines
```

---

## ⚠️ Disclaimer
*This chatbot provides medical information for educational purposes only and is not a substitute for professional clinical advice, diagnosis, or treatment.*
