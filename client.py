import asyncio
import sys

# We import the Client tools from the MCP library
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run():
    # 1. Define how to launch the server
    # We tell the client: "Run 'python server.py' to start the tool server"
    server_params = StdioServerParameters(
        command=sys.executable, # This finds your current 'python.exe'
        args=["server.py"],     # The script to run
        env=None                # Use default environment
    )

    print("🔌 Connecting to the Nutrition Tools Server...")

    # 2. Start the conversation
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # 3. Initialize
            await session.initialize()
            print("✅ Connected!")

            # 4. List Available Tools
            print("\n🔍 Asking server: 'What tools do you have?'")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"   - 🛠️  Tool Found: {tool.name} ({tool.description})")

            # 5. Call a Tool (Calculate BMI)
            print("\n❓ Asking server: 'Calculate BMI for 70kg, 1.75m'")
            result = await session.call_tool(
                name="calculate_bmi",
                arguments={"weight_kg": 70, "height_m": 1.75}
            )

            # 6. Show the Result
            print("🤖 Server Replied:")
            # The result comes back as a list of content blocks
            print(result.content[0].text)

if __name__ == "__main__":
    asyncio.run(run())