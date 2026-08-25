# 🏦 SBI Loan Intelligence Hub (RAG)

An end-to-end Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **LangChain**, and **OpenAI**. The system enables dynamic ingestion and querying of State Bank of India (SBI) loan policies, application forms, Most Important Terms & Conditions (MITC), and rate circulars.

---

### ✨ Features

* **Multi-Format Ingestion:** Process loan context from dynamic text input, PDF circulars, CSV rate sheets, JSON structures, and raw text files[cite: 1].
* **Configurable Chunking:** Adjust chunk sizes and overlap buffers via the UI using LangChain's `RecursiveCharacterTextSplitter`[cite: 1].
* **Vector Indexing:** Generate vector representations using `OpenAIEmbeddings` and store them in-memory via `ChromaDB`[cite: 1].
* **Strict Context-Grounded Q&A:** Synthesizes direct, hallucination-resistant answers with `gpt-4o-mini` strictly restricted to ingested document fragments[cite: 1].
* **Telemetry & Chunk Inspection:** Built-in UI to view chunk counts, active targets, and inspect raw retrieved context nodes[cite: 1].

---

### 🛠️ Tech Stack

* **Frontend UI:** Streamlit[cite: 1, 2]
* **Orchestration:** LangChain (`langchain`, `langchain-openai`, `langchain-community`, `langchain-classic`)[cite: 1, 2]
* **Embeddings & LLM:** OpenAI (`text-embedding-3-small`, `gpt-4o-mini`)[cite: 1]
* **Vector Store:** ChromaDB[cite: 1, 2]
* **Document Parsing:** PyPDF, Pandas[cite: 1, 2]

---

### 🚀 Getting Started

#### 1. Clone the repository
```bash
git clone [https://github.com/your-username/sbi-loan-intelligence-rag.git](https://github.com/your-username/sbi-loan-intelligence-rag.git)
cd sbi-loan-intelligence-rag
