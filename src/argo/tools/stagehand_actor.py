import os
from stagehand import Stagehand
from stagehand import StagehandConfig
from stagehand.schemas import AgentConfig, AgentExecuteOptions, AgentProvider
from dotenv import load_dotenv

load_dotenv()

def main():
    # Configure Stagehand
    config = StagehandConfig(
        env="LOCAL",
        model_name="claude-3-7-sonnet-latest",
        model_client_options={"apiKey": os.getenv("ANTHROPIC_API_KEY")}
    )

    # Initialize Stagehand
    stagehand = Stagehand(config=config)
    stagehand.init()
    print(f"Session created: {stagehand.session_id}")
    
    # Navigate to Google
    stagehand.page.goto("https://google.com/")
    
    # Configure the agent
    agent_config = AgentConfig(
        provider=AgentProvider.ANTHROPIC,
        model="claude-3-7-sonnet-latest",
        instructions="You are a helpful assistant that can use a web browser. Do not ask follow up questions, the user will trust your judgement.",
        options={"apiKey": os.getenv("ANTHROPIC_API_KEY")}
    )
    
    # Define execution options
    execute_options = AgentExecuteOptions(
        instruction="Search for 'latest AI news' and extract the titles of the first 3 results",
        max_steps=10,
        auto_screenshot=True
    )
    
    # Execute the agent task
    agent_result = stagehand.agent.execute(agent_config, execute_options)
    
    print(f"Agent execution result: {agent_result}")
    
    # Close the session
    stagehand.close()

if __name__ == "__main__":
    main()