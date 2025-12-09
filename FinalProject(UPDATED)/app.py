import streamlit as st
from agents.graph import app
from langchain_core.messages import HumanMessage
import os
import sys
import tempfile
from typing import List

# Ensure you have implemented this function in FinalProject/models/retriever.py
# It should accept a list of PDF file paths and return a LangChain Retriever object.
try:
    from models.retriever import get_rag_retriever_from_paths
except ImportError:
    st.error("Could not import get_rag_retriever_from_paths. Please check your models/retriever.py file.")
    sys.exit()


# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir) 

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="GraphQuery RAG Agent",
    page_icon="🤖",
    layout="wide"  
)

# --- CACHED FUNCTION TO BUILD RAG RETRIEVER ---
# Hashing trick: By passing file_paths (a list of strings), Streamlit hashes the list.
# The expensive function only runs if the list of paths changes (i.e., files are added/removed).
@st.cache_resource
def load_and_index_documents(file_paths: List[str]):
    """Loads documents and creates/returns a RAG retriever."""
    if not file_paths:
        return None
    
    with st.spinner(f"Indexing {len(file_paths)} PDF file(s)... This may take a moment."):
        try:
            # Calls the function from your models/retriever.py
            retriever = get_rag_retriever_from_paths(file_paths)
            st.success(f"Indexed {len(file_paths)} PDF file(s) successfully!")
            return retriever
        except Exception as e:
            st.error(f"Failed to index documents: {e}")
            return None

# --- SIDEBAR (Settings, Key, and Upload) ---
with st.sidebar:
    st.header("⚙️ Agent Settings")
    st.caption("Configure your LLM and Access Key.")
    
    # API Key Input
    api_key = st.text_input(
        "**Groq API Key (Required):**", 
        type="password",  
        help="Paste your private Groq API Key here. It is used only for this session.",
    )
    
    st.divider()
    
    # 1. FILE UPLOAD SECTION
    st.subheader("📚 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload your own PDFs for RAG context:", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    # 2. FILE SAVING & INDEXING LOGIC
    file_paths = []
    rag_retriever = None
    
    if uploaded_files:
        # Streamlit files are in memory; we must write them to a temporary file 
        # so LangChain's PyPDFLoader (which needs a file path) can read them.
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                # Write the file bytes to the temporary path
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)

            # 3. Build the retriever and cache it based on the list of paths
            # NOTE: We pass the list of temporary paths to the cached function.
            rag_retriever = load_and_index_documents(file_paths)

    else:
        # Clear the cache if no files are uploaded to ensure a clean state
        st.info("No documents uploaded. Only Wikipedia lookup is enabled.")
        load_and_index_documents.clear() # Clears the cache for this function
    

    st.divider()
    st.subheader("🛠️ Features")
    st.info(f"RAG (Document Context) status: {'**ENABLED**' if rag_retriever else 'DISABLED'}")
    st.info("Wikipedia Routing is always active.")
    st.text("MORE COMING SOON ⏱️")

# --- MAIN INTERFACE (Header) ---
st.markdown(
    """
    # 🧠 LangGraph Query Model 
    ### Multi-Source RAG Agent
    Ask a question related to your uploaded documents or general knowledge.
    """
)
st.divider()

# --- STATE INITIALIZATION ---
initial_state_base = {
    "documents": [],
    "source": "",
    "api_key": api_key, 
    # Pass the dynamically created retriever to the graph state
    "rag_retriever": rag_retriever 
}

# --- CHAT INPUT AND LOGIC ---
with st.form(key='query_form', clear_on_submit=True):
    user_query = st.text_input(
        "**Your Question:**", 
        placeholder="e.g., What is the significance of the military-industrial complex in Russia?",
        label_visibility="collapsed"
    )
    submit_button = st.form_submit_button(label='Ask the Agent 🚀')


# --- EXECUTION LOGIC ---

if submit_button and user_query:
    if not api_key:
        st.error("🔑 **Error:** Please enter your Groq API Key in the sidebar to run the query.")
        st.stop()
        
    st.info("🔄 **Querying the Agent...** Please wait.")
    
    # Prepare state
    initial_state = initial_state_base.copy()
    initial_state["messages"] = [HumanMessage(content=user_query)]
    
    with st.spinner('Thinking... Routing and Retrieving Context...'):
        try:
            response = app.invoke(initial_state)
            
            # --- Output Display ---
            final_message = response["messages"][-1].content
            
            st.success("✅ **Agent Response:**")
            st.markdown(final_message)
            st.divider()
            
            # Optional: Show debug info
            with st.expander("🔍 **Debug Info (Agent Flow)**"):
                st.write(f"**Final Source:** {response.get('source', 'Unknown')}")
                if 'documents' in response and response['documents']:
                    st.write(f"**Retrieved Documents:** {len(response['documents'])} chunks used.")
                
        except Exception as e:
            st.error("❌ **Agent Failure:** An error occurred during execution.")
            st.exception(e)
            
elif not user_query and not api_key:
    st.markdown("👋 Start by entering your **Groq API Key** in the sidebar and asking a question above!")