import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma 

# Load OpenAI keys from your local .env file
load_dotenv()

st.set_page_config(
    page_title="SBI Loan Intelligence Hub - RAG",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 🎨 UI & COSMETIC STYLING: Premium Modern Theme Injection
# =========================================================
st.markdown(
    """
    <style>
        /* Sidebar layout customization */
        [data-testid="stSidebar"] {
            min-width: 35rem !important;
            max-width: 38rem !important;
            background-color: #1e222b;
        }
        
        /* Main Application Background Polish */
        .stApp {
            background-color: #0e1117;
        }
        
        /* Custom Dashboard Style Metric Cards */
        .metric-card {
            background-color: #171a21;
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #00b4d8;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
            margin-bottom: 20px;
        }
        .metric-title {
            font-size: 0.85rem;
            color: #8a92a6;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #ffffff;
        }

        /* Modernized Dividers */
        .custom-hr {
            margin: 25px 0;
            border: 0;
            height: 1px;
            background: linear-gradient(to right, rgba(0, 180, 216, 0.5), rgba(0, 0, 0, 0));
        }
        
        /* Subtle adjustments for text areas & inputs inside dark UI */
        .stTextArea textarea {
            background-color: #171a21 !important;
            color: #e2e8f0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("SBI Loan Intelligence Assistant Using RAG🏦")
st.caption("⚡ A premium, RAG-grounded engine for tracking SBI interest terms, loan conditions, and eligibility rules.")

# =========================================================
# LEFT SIDEBAR: Pipeline Mapping Control Dashboard
# =========================================================
with st.sidebar:
    st.header("📋 Pipeline Setup")
    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
    
    # 1. Source Documents
    st.subheader("1. Source Documents")
    uploaded_files = st.file_uploader(
        "Upload contextual references (MITC, Circulars, Application Forms):", 
        type=["csv", "txt", "pdf", "json"],
        accept_multiple_files=True
    )
    
    pasted_text = st.text_area(
        "Or enter text dynamically:",
        height=140,
        placeholder="Paste interest rate matrices, terms conditions, or custom loan schemas..."
    )
    
    # Global structured list to hold isolated raw Document objects
    docs_list = []
    selected_loan_context = None
    
    if pasted_text:
        docs_list.append(Document(page_content=pasted_text, metadata={"source": "Manual Dynamic Paste"}))
        
    if uploaded_files:
        st.toast(f"Successfully staged {len(uploaded_files)} loan resource files!", icon="📂")
        
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            if file_type == "csv":
                df = pd.read_csv(uploaded_file)
                unique_loans = sorted(df['loan_type'].dropna().unique()) if 'loan_type' in df.columns else sorted(df.iloc[:,0].dropna().unique())
                selected_loan_context = st.selectbox(f"🎯 Target Product ({uploaded_file.name}):", unique_loans)
                
                if selected_loan_context:
                    col_name = 'loan_type' if 'loan_type' in df.columns else df.columns[0]
                    row = df[df[col_name] == selected_loan_context].iloc[0]
                    docs_list.append(Document(page_content=row.to_string(), metadata={"source": f"{uploaded_file.name}_{selected_loan_context}"}))
            
            elif file_type == "json":
                try:
                    json_data = json.load(uploaded_file)
                    docs_list.append(Document(page_content=json.dumps(json_data), metadata={"source": uploaded_file.name}))
                except Exception as e:
                    st.error(f"Error parsing JSON configuration: {e}")
                    
            elif file_type == "txt":
                raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
                docs_list.append(Document(page_content=raw_text, metadata={"source": uploaded_file.name}))

            elif file_type == "pdf":
                try:
                    pdf_reader = PdfReader(uploaded_file)
                    # Extract page by page to force vector fragment variance
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            docs_list.append(Document(
                                page_content=page_text, 
                                metadata={"source": uploaded_file.name, "page": page_num + 1}
                            ))
                except Exception as e:
                    st.error(f"Failed to read PDF payload: {e}")

    # 2. Parsing & Splitting Strategies
    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
    st.subheader("2. Parsing & Splitting Strategies")
    chunk_size = st.slider("Max Chunk Size", 50, 2000, 500)
    chunk_overlap = st.slider("Overlap Buffer Window", 0, 400, 80)

    # 3. Embedding Model Selection
    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
    st.subheader("3. Neural Vectorizer Mapping")
    embedding_selection = st.selectbox("Text Vector Model:", ["openai/text-embedding-3-small", "local/in-memory-embeddings"])

    # 4. Vector Store Settings
    st.subheader("4. Vector Index Architecture")
    vector_store_selection = st.selectbox("Storage Index Engine:", ["ChromaDB (Local Cache)", "FAISS Index"])

    # 5. Pipeline Query Target
    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
    st.subheader("5. Ask Assistant")
    user_query = st.text_input(
        "Query loan properties:", 
        placeholder="e.g., What is the penal fee for late construction completion under the Home Loan MITC?"
    )
    
    st.write("")
    run_pipeline = st.button("Run Engine Layer ⚡", type="primary", use_container_width=True)


# =========================================================
# RIGHT PANEL: Elegant Response Showcase Box
# =========================================================
main_response_box = st.container(border=True)

with main_response_box:
    st.subheader("🎯 System Response Window")
    
    if run_pipeline and docs_list and user_query:
        with st.spinner("Executing structural vector processing matrix..."):
            
            # --- STAGE 1 & 2: Chunking Isolated Document Array ---
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            final_chunks = text_splitter.split_documents(docs_list)
            
            # Styled Custom Metric Display Showcase
            source_label = selected_loan_context if selected_loan_context else f"Data Pool ({len(uploaded_files)} files)"
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Active Loan Context Target</div>
                    <div class="metric-value">{str(source_label)[:28]}...</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Vector Graph Fragments</div>
                    <div class="metric-value">{len(final_chunks)} Chunks</div>
                </div>
                """, unsafe_allow_html=True)
            
            # --- STAGE 3 & 4: Embedding / Index Ingest ---
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(final_chunks, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": min(len(final_chunks), 5)})
            
            # --- STAGE 5: Synthesis Pipeline Execution ---
            system_prompt = (
                "You are an expert, direct SBI Banking Loan officer and financial advisor.\n"
                "Your objective is to provide simple, crisp, clear, and user-friendly answers about loans.\n"
                "Avoid unnecessary boilerplate text or noise so that users can instantly understand the answer.\n\n"
                "CRITICAL CONSTRAINT:\n"
                "Rely strictly on the verified custom context document fragments provided below to answer the query.\n"
                "If the data needed to answer the user query cannot be found within the provided context fragments, reply honestly with:\n"
                "'I don't have that specific data in my current database.' Do not attempt to guess or hallucinate any financial details.\n\n"
                "Context Fragments:\n"
                "{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)  # Low temp ensures strict context compliance
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            response = rag_chain.invoke({"input": user_query})
            
            # Modern Clean Results Block
            st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
            st.markdown("### 📝 Strategic Loan Response")
            st.info(response["answer"])
            
            # Telemetry Analytics Window
            st.write("")
            with st.expander("🔍 System Telemetry: Inspect Isolated Datastore Chunks"):
                for idx, chunk in enumerate(final_chunks):
                    source_info = chunk.metadata.get('source', 'Unknown')
                    page_info = f" | Page: {chunk.metadata.get('page')}" if 'page' in chunk.metadata else ""
                    st.markdown(f"**Chunk ID Node #{idx+1} ({source_info}{page_info})**")
                    st.code(chunk.page_content, language="text")
                    
    elif run_pipeline and not docs_list:
        st.warning("⚠️ Operational constraint: Please load loan documentation variables or PDFs in Step 1 before execution.")
    else:
        st.info("💡 Adjust architectural properties via the side configuration dashboard, then activate **Run Engine Layer ⚡**.")