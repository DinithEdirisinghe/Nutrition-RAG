import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load keys
load_dotenv()

# 2. Load the Web Page
# We are loading an article about Avocados
url = "https://www.healthline.com/nutrition/12-proven-benefits-of-avocado"
print(f"🌐 Visiting: {url}...")

loader = WebBaseLoader(url)
data = loader.load()
print("✅ Page loaded!")

# 3. Split the text
# Web pages are long, so we use a smarter splitter that respects paragraph breaks
print("✂️  Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(data)
print(f"   -> Created {len(chunks)} text chunks.")

# 4. Embed and Save
print("🔢 Converting to numbers...")
embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

print("💾 Saving to Vector Database...")
# We are adding this to your EXISTING database folder
db = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_function, 
    persist_directory="./chroma_db"
)

print("🎉 Success! Your AI now knows about Avocados.")