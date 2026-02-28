import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
from dotenv import load_dotenv

# 1. Load your keys from .env
def _find_env_file() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find .env in this directory or any parent directory")

load_dotenv(dotenv_path=_find_env_file())

supabase_url = os.getenv("SUPABASE_URL")
supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")

if not supabase_url or not supabase_service_key:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env")

if supabase_service_key.startswith("sb_publishable_"):
    raise ValueError(
        "SUPABASE_SERVICE_KEY is set to a publishable key. Use your Supabase secret/service_role key for ingestion."
    )

supabase = create_client(supabase_url, supabase_service_key)

# 2. Load the textbook
print("Loading PDF...")
loader = PyPDFLoader("ck12_basic_algebraFree.pdf") # Ensure your PDF is in the same folder
pages = loader.load()

# 3. Chop the text into small, 1000-character chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)
# 3. Chop the text into small, 1000-character chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)

# --- NEW: CLEANING STEP ---
print("Cleaning null characters from chunks...")
for chunk in chunks:
    # Remove the \u0000 characters that PostgreSQL hates
    chunk.page_content = chunk.page_content.replace("\x00", "")
# ---------------------------

# 4. Create the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# 4. Create the embedding model (converts text to vector numbers)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Upload everything to Supabase
print("Uploading to Supabase... this might take a minute.")
vector_store = SupabaseVectorStore.from_documents(
    chunks,
    embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents"
)
print("Done! The textbook is in the database.")