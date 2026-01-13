from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load .env file
load_dotenv()

# Load API keys from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Validate API keys
if not isinstance(PINECONE_API_KEY, str) or not PINECONE_API_KEY.strip():
    raise ValueError("PINECONE_API_KEY is missing or empty. Check your .env file.")

if not isinstance(OPENROUTER_API_KEY, str) or not OPENROUTER_API_KEY.strip():
    raise ValueError("OPENROUTER_API_KEY is missing or empty. Check your .env file.")

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

print("✅ API keys loaded successfully!")

# -----------------------
# Load and process PDFs
# -----------------------
extracted_data = load_pdf_file(data='data/')
filtered_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filtered_data)

# -----------------------
# Load embeddings
# -----------------------
embeddings = download_hugging_face_embeddings()

# -----------------------
# Connect to Pinecone
# -----------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"  # Change if needed

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# -----------------------
# Create vector store
# -----------------------
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)

print(f"✅ Index '{index_name}' created/loaded successfully with {len(text_chunks)} chunks.")
