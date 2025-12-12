import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load keys
load_dotenv()

# 2. Setup Embedding (The Translator)
print("🔌 Connecting to database...")
embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 3. Load Database
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
retriever = db.as_retriever(search_kwargs={"k": 1})

# 4. Setup Brain (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

# 5. Create the Prompt Template
# We tell the AI: "Here is the context (facts). Answer the question based ONLY on this."
template = """
You are a helpful nutrition assistant. Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 6. Build the Modern Chain (LCEL)
# This says: Take question -> Find Facts -> Put into Prompt -> Send to Gemini -> Read Answer
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Ask
question = "How do avocados help with eye health?"
print(f"❓ Asking: {question}\n")

try:
    response = rag_chain.invoke(question)
    print("🤖 Gemini Answer:")
    print(response)
except Exception as e:
    print(f"❌ Error: {e}")