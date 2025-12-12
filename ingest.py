import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load environment variables
load_dotenv()

# 2. Load the text file
print("📚 Loading data...")
loader = TextLoader("./data/facts.txt")
document = loader.load()

# 3. Split text into chunks (AI can't read whole books at once)
print("✂️  Splitting text...")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(document)

# 4. Initialize the Embedding Model (Turns text into numbers)
print("🔢 Converting text to numbers...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 5. Create and Save the Database
print("💾 Saving to Vector Database...")
db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print(f"✅ Success! Saved {len(chunks)} chunks to the database.")