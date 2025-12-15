import asyncio
import sys
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Setup the Brain
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

async def run_agent():
    # 2. Connect to the MCP Server
    server_params = StdioServerParameters(
        command=sys.executable, 
        args=["server.py"], 
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 3. Discover Tools (The AI learns what it can do)
            tools_list = await session.list_tools()
            tool_descriptions = "\n".join(
                [f"- {t.name}: {t.description}" for t in tools_list.tools]
            )
            print(f"🛠️  Agent found these tools:\n{tool_descriptions}\n")

            # 4. The User Question
            user_question = "I weigh 85 kg and I am 1.8 meters tall. Is my weight healthy?"
            print(f"❓ User asks: '{user_question}'")

            # 5. The Agent Loop (ReAct)
            # We explain the rules to Gemini: "Don't guess. Use the tool."
            system_prompt = f"""
            You are a smart nutrition assistant. You have access to these tools:
            {tool_descriptions}

            To use a tool, reply EXACTLY in this format:
            ACTION: tool_name | arguments_in_json

            Example: ACTION: calculate_bmi | {{"weight_kg": 70, "height_m": 1.75}}

            If you have the answer, reply:
            FINAL ANSWER: [your response]

            Question: {user_question}
            """

            # -- Step 1: Brain decides what to do --
            response = llm.invoke(system_prompt).content
            print(f"\n🧠 Agent thought: {response}")

            # -- Step 2: Check if it wants to take action --
            if "ACTION:" in response:
                # Parse the request (extract tool name and args)
                action_part = response.split("ACTION:")[1].strip()
                tool_name, args_str = action_part.split("|")
                import json
                args = json.loads(args_str)

                print(f"🏃 Agent is running tool: {tool_name} with {args}...")

                # -- Step 3: Run the tool via MCP --
                tool_result = await session.call_tool(tool_name.strip(), args)
                result_text = tool_result.content[0].text
                print(f"📊 Tool Output: {result_text}")

                # -- Step 4: Give the result back to the Brain --
                final_prompt = f"""
                The tool returned this result: {result_text}

                Now answer the original question: {user_question}
                """
                final_answer = llm.invoke(final_prompt).content
                print(f"\n🤖 FINAL ANSWER:\n{final_answer}")
            else:
                print(f"\n🤖 FINAL ANSWER:\n{response}")

if __name__ == "__main__":
    asyncio.run(run_agent())

    # 6. Exit