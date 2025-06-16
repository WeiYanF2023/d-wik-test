from contextlib import AsyncExitStack
from typing import Tuple, Union
from google.adk.agents import (
    LlmAgent
)
from google.adk.models import BaseLlm
from google.adk.tools.agent_tool import AgentTool

from argo.tools import get_browser_tools

DEFAULT_BROWSER_ACTOR_SYSTEM_PROMPT = """You are a helpful assistant that can use a web browser via playwright tools. 
You will receive specific requests and are invited to try to fulfill them using your tools. 
If you complete the request, please output the status at the end and tell the result; 
If you encounter any difficulties in executing it that prevent the request from being completed, please output it at the end and tell the specifics of the difficulties;
If you think the request is not clear enough, you can ask for a refinement of request."""

DEFAULT_BROWSER_ACTOR_DESCRIPTION = """Browser Actor is an actor to use browser with some intelligence of its own. It understands natural language action commands and directly manipulates the web page to execute the corresponding commands. You should break down the complex task you want to perform into multiple simple steps and then request the Actor to perform them in turn. Here's an example string of requests

Request: Please visit Amazon.com and tell me what's on the page.
Actor: [Feedback]
Request:Please enter laptop in the search box and click search.
Actor: [Feedback]
Request: Please sort the products by sales.
Actor: [Feedback]
Request: Find the top three selling products and what brands are they?
Actor: [Feedback]
Request: Please send me a link to the second highest selling product.
Actor: [Feedback]

Please refer to the above examples to make the best use of this tool.
"""

async def get_browser_actor(
    model: Union[str, BaseLlm],
    name: str = "Browser_Actor",
    browser: str = "firefox",
    system_prompt: str = DEFAULT_BROWSER_ACTOR_SYSTEM_PROMPT,
) -> Tuple[AgentTool, AsyncExitStack]:
    """
    Get a browser actor with the specified model and browser.
    Args:
        model (Union[str, BaseLlm]): The model to use for the actor.
        name (str): The name of the actor (default: "Browser_Actor").
        browser (str): The browser to use (default: "firefox").
        system_prompt (str): System prompt for the actor.
    Returns:
        Tuple[AgentTool, AsyncExitStack]: A tuple containing the browser actor and an exit stack for cleanup.
    """
    
    browser_tools, exit_stack = await get_browser_tools(browser=browser)
    browser_actor = LlmAgent(
        name=name,
        description=DEFAULT_BROWSER_ACTOR_DESCRIPTION,
        model=model,
        tools=browser_tools,
        instruction=system_prompt,
    )
    return AgentTool(agent=browser_actor), exit_stack