import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# 1. Load the secret keys
load_dotenv()

# 2. Wake up the Embedding Function
# (We need this to understand the numbers in the database)
embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 3. Load the Database we built earlier
# We tell it: "Look in the ./chroma_db folder"
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# 4. Wake up the Brain (LLM)
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

# 5. Connect Brain to Database (The RAG Chain)
# chain_type="stuff" means "Stuff all the found facts into the prompt"
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 1})
)

# 6. Ask the Question
question = "What happens if I don't get enough Magnesium?"
print(f"❓ Asking: {question}")

response = qa_chain.invoke({"query": question})

print("\n🤖 Answer based on your facts:")
print(response["result"])