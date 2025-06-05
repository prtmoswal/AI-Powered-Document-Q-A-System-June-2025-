import os
import io
from docx import Document
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile # Needed for saving uploaded files temporarily

# --- Configuration for Chunking and Models ---
CHUNK_SIZE_WORDS = 500 # Words per chunk (more semantic than characters)
CHUNK_OVERLAP_WORDS = 50 # Overlap in words

# Embedding model name
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Local LLM for Extractive QA model name
LOCAL_LLM_MODEL_NAME = "distilbert-base-uncased-distilled-squad"

# Global variables for models (initialized once)
embedding_model_instance = None
qa_pipeline_instance = None

# --- Model Initialization ---
def initialize_models():
    """Initializes embedding and local LLM models once."""
    global embedding_model_instance, qa_pipeline_instance

    if embedding_model_instance is None:
        print("Initializing embedding model...")
        embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model initialized.")

    if qa_pipeline_instance is None:
        print(f"Initializing local LLM (for extractive QA): {LOCAL_LLM_MODEL_NAME}...")
        try:
            qa_pipeline_instance = pipeline(
                "question-answering",
                model=LOCAL_LLM_MODEL_NAME,
                tokenizer=LOCAL_LLM_MODEL_NAME,
                device="cpu" # Use "cuda" if you have a compatible GPU
            )
            print("Device set to use cpu") # Inform the user about the device
            print("Local LLM (extractive QA) initialized.")
        except Exception as e:
            print(f"Error initializing local LLM: {e}")
            qa_pipeline_instance = None

# --- Document Processing Functions ---

def extract_text_from_docx(docx_path: str) -> str:
    """Extracts text from a .docx file given its path."""
    doc = Document(docx_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    raw_text = "\n".join(full_text)

    # Basic cleaning
    cleaned_text = raw_text.replace("[Image 1]", "").replace("[Image 2]", "").strip()
    # Remove empty lines that might result from cleaning
    cleaned_text = "\n".join([line for line in cleaned_text.splitlines() if line.strip()])
    return cleaned_text

def chunk_text_by_words(text: str, chunk_size_words: int, chunk_overlap_words: int) -> list[str]:
    """Splits text into overlapping chunks based on words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size_words - chunk_overlap_words):
        chunk = " ".join(words[i : i + chunk_size_words])
        if chunk: # Ensure chunk is not empty
            chunks.append(chunk)
    return chunks

# This function will now accept a docx_path and return the processed data
def load_and_process_document(docx_path: str):
    """
    Loads text from a .docx file, chunks it, and creates a FAISS index.
    Returns (faiss_index, list_of_chunks_text).
    """
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"Document not found at: {docx_path}")

    initialize_models() # Ensure models are initialized

    print(f"Processing document from: {os.path.basename(docx_path)}")
    cleaned_text = extract_text_from_docx(docx_path)
    document_chunks_list = chunk_text_by_words(cleaned_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)

    if not document_chunks_list:
        print("No content extracted or chunks created from the document.")
        return None, []

    print(f"Creating embeddings for {len(document_chunks_list)} chunks...")
    chunk_embeddings = embedding_model_instance.encode(document_chunks_list, show_progress_bar=True)
    
    # Ensure embeddings are float32 for FAISS
    document_embeddings = np.array(chunk_embeddings).astype('float32')

    # Create FAISS index
    dimension = document_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension) # L2 distance for similarity
    faiss_index.add(document_embeddings)
    print("FAISS index created.")

    return faiss_index, document_chunks_list

# --- QA and Summary Functions ---

def get_document_answer(question: str, faiss_index, document_chunks: list[str], top_k_chunks: int = 3) -> str:
    """
    Retrieves relevant chunks from the provided index/chunks
    and generates an answer using the local extractive LLM.
    """
    if embedding_model_instance is None or qa_pipeline_instance is None:
        return "Core AI models not initialized. Please ensure setup is complete."

    if faiss_index is None or not document_chunks:
        return "No document is loaded or processed. Please upload a document first."

    # 1. Embed the question
    print(f"Searching for relevant document chunks for question: {question}")
    question_embedding = embedding_model_instance.encode([question]).astype('float32')

    # 2. Search FAISS index for relevant chunks
    D, I = faiss_index.search(question_embedding, top_k_chunks)
    relevant_contexts = [document_chunks[i] for i in I[0]]
    print(f"Found {len(relevant_contexts)} relevant contexts.")

    print("\n--- Relevant Contexts Found ---")
    for i, context_text in enumerate(relevant_contexts):
        # Print first 200 chars of context for debugging
        print(f"Context {i+1}:\n{context_text[:200]}...\n")
    print("--------------------------------\n")

    # 3. Prepare context for LLM
    context = "\n\n".join(relevant_contexts)

    if not context.strip():
        return "No relevant information found in the document for your question based on current search."

    # 4. Use the extractive QA pipeline
    qa_input = {
        'question': question,
        'context': context
    }
    print("Generating answer using local LLM pipeline...")
    try:
        answer = qa_pipeline_instance(qa_input)
        return answer.get('answer', 'Could not find a specific answer in the document.')
    except Exception as e:
        print(f"Error during local extractive QA: {e}")
        return f"Error generating answer from local LLM: {e}"


def get_local_document_summary(document_chunks: list[str], doc_name: str) -> str:
    """Provides a basic summary based on document properties and word count."""
    if not document_chunks:
        return "Document content not loaded. Cannot provide a summary."

    total_words = sum(len(chunk.split()) for chunk in document_chunks)
    num_chunks = len(document_chunks)

    summary_text = (
        f"This document, '{doc_name}', contains approximately {total_words:,} words, organized into {num_chunks} searchable sections. "
        "It appears to be a technical or regulatory document (based on its .docx format). "
        "You can now ask specific questions about its content, and the system will try to extract relevant information."
    )
    return summary_text