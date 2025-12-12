import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

print("🔌 Connecting to database...")
embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# 1. RETRIEVAL STEP
# We ask the database for the docs DIRECTLY so we can print them
retriever = db.as_retriever(search_kwargs={"k": 5})
question = "How do avocados help with eye health?"

print(f"❓ Asking: {question}\n")
print("🔍 Retrieving documents...")
docs = retriever.invoke(question)

# --- DEBUG PRINT ---
print(f"📄 Found {len(docs)} chunks. Here is the first one:")
print("--------------------------------------------------")
print(docs[0].page_content[:500]) # Print only first 500 characters
print("... (truncated)")
print("--------------------------------------------------\n")
# -------------------

# 2. GENERATION STEP
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

template = """
You are a helpful nutrition assistant. Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# We manually pass the 'docs' we just found into the chain
chain = prompt | llm | StrOutputParser()

response = chain.invoke({"context": docs, "question": question})

print("🤖 Gemini Answer:")
print(response)