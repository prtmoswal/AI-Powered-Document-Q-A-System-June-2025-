# app.py
import streamlit as st
import os
import tempfile # Needed for saving uploaded file temporarily
from sentence_transformers import SentenceTransformer # Used for cache_resource hash_funcs

# Import updated backend functions for document processing, QA, and local summary
from backend import get_document_answer, load_and_process_document, initialize_models, get_local_document_summary

# --- Streamlit App Configuration ---
st.set_page_config(page_title="AI Document Q&A", layout="wide")

st.title("🏛️ AI-Powered Document Q&A")
st.markdown("Upload a document, then ask questions or request a summary!")

# --- Initialize Core AI Models ---
# This function uses st.cache_resource to ensure embedding and local LLM models
# are loaded only once across the entire app's lifecycle, improving performance.
@st.cache_resource(hash_funcs={SentenceTransformer: lambda _: None, type(None): lambda _: None})
def setup_core_models():
    """Initializes global models (embedding and local LLM) in backend."""
    initialize_models()
    return True

# Run model setup on app start
if setup_core_models():
    st.sidebar.success("Core AI Models Initialized!")
else:
    st.sidebar.error("Failed to initialize core AI models. Please check your setup.")
    st.stop() # Stop the app if models can't be loaded

# --- Document Upload Section ---
st.sidebar.subheader("1. Upload Your Document")
uploaded_file = st.sidebar.file_uploader("Upload a **.docx** file", type=["docx"], key="doc_uploader")

# --- Streamlit Session State for Document Data ---
# These variables will store the processed document's index and chunks,
# making them persistent across app reruns for the current user session.
if 'document_index' not in st.session_state:
    st.session_state['document_index'] = None
if 'document_chunks' not in st.session_state:
    st.session_state['document_chunks'] = []
if 'current_document_name' not in st.session_state:
    st.session_state['current_document_name'] = "None"
if 'last_uploaded_file_name' not in st.session_state: # To detect if a new file is uploaded
    st.session_state['last_uploaded_file_name'] = None
if 'last_uploaded_file_size' not in st.session_state: # To detect if a new file is uploaded
    st.session_state['last_uploaded_file_size'] = None


# --- Function to Process and Cache Uploaded Document ---
# This function is also cached. It will only re-run if the *content* of the uploaded file changes.
# hash_funcs are crucial for correctly hashing the file object.
@st.cache_resource(hash_funcs={type(st.runtime.uploaded_file_manager.UploadedFile): lambda x: x.read(), "_io.BufferedReader": id, "tempfile._TemporaryDirectoryGuard": id})
def process_uploaded_document_cached(uploaded_file_obj, file_name):
    """
    Processes the uploaded document, creates embeddings, and returns FAISS index and chunks.
    This function leverages Streamlit's cache_resource to avoid re-processing the same file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Streamlit's uploaded_file_obj is like a file, but needs to be reset for reading
        uploaded_file_obj.seek(0)
        temp_file_path = os.path.join(tmpdir, file_name)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file_obj.read())

        # Call the backend function to load and process the document
        doc_index, doc_chunks = load_and_process_document(temp_file_path)
        return doc_index, doc_chunks

# --- Logic to Handle Document Upload ---
if uploaded_file is not None:
    # Check if a new file has been uploaded or if the current document is not set
    is_new_file = (
        st.session_state['current_document_name'] != uploaded_file.name or
        st.session_state['last_uploaded_file_size'] != uploaded_file.size or
        st.session_state['document_index'] is None # If no document is currently loaded
    )

    if is_new_file:
        st.sidebar.info(f"Processing new document: **{uploaded_file.name}**...")
        try:
            # Process the new document (this will use or update the cache)
            doc_index, doc_chunks = process_uploaded_document_cached(uploaded_file, uploaded_file.name)

            # Store the processed data in session state
            st.session_state['document_index'] = doc_index
            st.session_state['document_chunks'] = doc_chunks
            st.session_state['current_document_name'] = uploaded_file.name
            st.session_state['last_uploaded_file_name'] = uploaded_file.name # Update for next check
            st.session_state['last_uploaded_file_size'] = uploaded_file.size # Update for next check

            st.sidebar.success(f"Document **'{uploaded_file.name}'** loaded and indexed successfully!")
            st.rerun() # Rerun the app to update the main content with the new document status
        except Exception as e:
            st.sidebar.error(f"Error processing document: {e}")
            # Reset state on error
            st.session_state['document_index'] = None
            st.session_state['document_chunks'] = []
            st.session_state['current_document_name'] = "None"
            st.session_state['last_uploaded_file_name'] = None
            st.session_state['last_uploaded_file_size'] = None

# Display the status of the currently active document
if st.session_state['document_index'] is None:
    st.warning("Please upload a **.docx** document to start asking questions.")
else:
    st.info(f"Currently active document: **{st.session_state['current_document_name']}**")


# --- User Input Section ---
st.subheader("2. Ask Your Question")
user_question = st.text_input("Enter your question here:", key="user_question")

if st.button("Get Answer"):
    if st.session_state['document_index'] is None:
        st.error("No document is loaded. Please upload a **.docx** file first.")
    elif user_question:
        with st.spinner("Thinking..."):
            lower_user_question = user_question.lower().strip()

            # --- Detect if the user is asking for a summary ---
            is_summarize_request = any(keyword in lower_user_question for keyword in [
                "summarize", "summary", "give me an overview", "what is this document about",
                "what's in this document", "document overview", "contents of this document"
            ])

            if is_summarize_request:
                st.info("Generating a basic local summary of the document...")
                # Call the local summarization function, passing current document data
                answer = get_local_document_summary(
                    st.session_state['document_chunks'],
                    st.session_state['current_document_name']
                )
                st.subheader("Document Summary (Local):")
                st.success(answer)
                st.caption("*(This is a basic, locally generated summary based on document properties and a general description.)*")
            else:
                # All other questions go to the document-specific extractive QA
                answer = get_document_answer(
                    user_question,
                    st.session_state['document_index'], # Pass the FAISS index
                    st.session_state['document_chunks'] # Pass the document chunks
                )
                st.subheader("Answer from Document (Extractive):")
                st.info(answer)
                st.caption("*(Answer based on the currently loaded document. If the answer is not precise, try rephrasing.)*")
    else:
        st.warning("Please enter a question.")

# --- View Document Content Section ---
st.markdown("---")
with st.expander("📖 **View Uploaded Document Content**"):
    if st.session_state['document_chunks']:
        full_document_text = "\n\n".join(st.session_state['document_chunks'])
        st.text_area(f"Content of '{st.session_state['current_document_name']}'",
                     value=full_document_text,
                     height=500, # Adjust height as needed
                     key="document_viewer")
    else:
        st.info("Upload a document first to view its content here.")

# --- How it Works Section ---
st.markdown("---") # Visual separator
st.write("### How it Works:") # A subheader
st.markdown("""
- **Upload Document:** Upload a **.docx** file. The app will process its content using **local AI models** (Sentence Transformer for embeddings, DistilBERT for QA) to create numerical representations and a searchable index. This processing is cached for speed.
- **Document-Specific Questions:** Your question is compared to the content of the **currently active document**. Relevant sections are retrieved, and the **local AI model** generates an answer *based only on that document's text* by extracting a relevant span.
- **Document Summarization:** If your question implies a summary (e.g., "summarize this document"), a **basic local summary** is provided. This summary is based on the document's word count and general characteristics, *not* a deep generative summary.
""")

# --- Disclaimer Section ---
st.markdown("---") # Visual separator
st.write("#### Disclaimer:") # A subheader
st.caption("This AI tool ('AI Document Q&A') is for informational and educational purposes only. It relies on AI models, which can sometimes make errors or misinterpret text. It does NOT provide legal advice, financial advice, or official interpretations of government policies. Always refer to the original government document or consult official government authorities for definitive information.")