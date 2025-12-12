import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load the secret key from the .env file
load_dotenv()

# 2. Check if key exists
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found. Did you create the .env file?")
    exit()

print("✅ Key found! Connecting to Gemini...")

# 3. Initialize the Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

# 4. Ask a simple question
try:
    response = llm.invoke("What are the three macronutrients? Answer in one sentence.")
    print("\n🤖 Gemini says:")
    print(response.content)
    print("\n🎉 SUCCESS: Your AI is connected and ready to work!")
except Exception as e:
    print(f"\n❌ Connection failed: {e}")