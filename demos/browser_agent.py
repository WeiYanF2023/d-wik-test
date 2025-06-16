import json
from argo.actors import get_browser_actor

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME = "Browser_Agent_App"
USER_ID = "test_user"
SESSION_ID = "test_session"

session_service = InMemorySessionService()
session_service.create_session(
    app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
)


async def call_agent_and_print(
    runner_instance: Runner, agent_instance: LlmAgent, session_id: str, query: str
):
    """Sends a query to the specified agent/runner and prints results."""
    print(f"\n>>> Calling Agent: '{agent_instance.name}' | Query: {query}")

    user_content = types.Content(role="user", parts=[types.Part(text=query)])

    final_response_content = "No final response received."
    async for event in runner_instance.run_async(
        user_id=USER_ID, session_id=session_id, new_message=user_content
    ):
        # print(f"Event: {event.type}, Author: {event.author}") # Uncomment for detailed logging
        if event.is_final_response() and event.content and event.content.parts:
            # For output_schema, the content is the JSON string itself
            final_response_content = event.content.parts[0].text

    print(f"<<< Agent '{agent_instance.name}' Response: {final_response_content}")

    current_session = session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    stored_output = current_session.state.get(agent_instance.output_key)

    # Pretty print if the stored output looks like JSON (likely from output_schema)
    print(f"--- Session State ['{agent_instance.output_key}']: ", end="")
    try:
        # Attempt to parse and pretty print if it's JSON
        parsed_output = json.loads(stored_output)
        print(json.dumps(parsed_output, indent=2))
    except (json.JSONDecodeError, TypeError):
        # Otherwise, print as string
        print(stored_output)
    print("-" * 30)


async def main():
    # Initialize the browser actor with the default system prompt
    browser_actor, exit_stack = await get_browser_actor(
        model="gemini-2.5-flash-preview-04-17",
    )

    browser_agent = LlmAgent(
        name="Browser_Agent",
        model="gemini-2.5-pro-preview-03-25",
        tools=[browser_actor],
        instruction="You are a helpful assistant that can interact with a web browser actor. Please use these tools to assist with user's request.",
    )

    browser_runner = Runner(
        app_name=APP_NAME,
        agent=browser_agent,
        session_service=session_service,
    )

    await call_agent_and_print(
        browser_runner,
        browser_agent,
        session_id=SESSION_ID,
        query="In NASA’s Astronomy Picture of the Day on 2006 January 21, two astronauts are visible, with one appearing much smaller than the other. As of August 2023, out of the astronauts in the NASA Astronaut Group that the smaller astronaut was a member of, which one spent the least time in space, and how many minutes did he spend in space, rounded to the nearest minute? Exclude any astronauts who did not spend any time in space. Give the last name of the astronaut, separated from the number of minutes by a semicolon. Use commas as thousands separators in the number of minutes.",
    )

    print("Closing MCP server connection...")
    await exit_stack.aclose()
    print("Cleanup complete.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
