import os
import json
import asyncio
import logging
import re
import mimetypes
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

import datasets
from datasets import Dataset

# from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.tools import google_search
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams, StdioServerParameters
from google.adk.agents import (
    LlmAgent,
)
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Import custom toolkits
from argo.tools import (
    general_search_tool,
    google_scholar_tool,
    get_content_from_url_tool,
    get_content_from_path_tool,
    python_execution_tool,
    unzip_tool,
    search_wikipedia_tool,
    get_wikipedia_page_tool,
    get_wikipedia_categories_tool,
    wolfram_alpha_tool,
    youtube_understanding_tool,
    google_maps_tools,
    get_browser_tools
)

from argo.actors import get_browser_actor
from argo.workflows import get_group_chat_agent
from argo.models import ArgoLiteLLM, Claude

# --- Configuration ---
RESULTS_DIR = "results"
RESULT_FILE_VALIDATION = os.path.join(RESULTS_DIR, "gaia_results_validation_argo.jsonl")
TRACE_FILE_VALIDATION = os.path.join(RESULTS_DIR, "gaia_trace_validation_argo.txt")
APP_NAME = "GAIA_Evaluation"
SUPPORTED_MIMES = [
    'application/pdf',
    'application/x-javascript',
    'text/javascript',
    'application/x-python',
    'text/x-python',
    'text/plain',
    'text/html',
    'text/css',
    'text/md',
    'text/csv',
    'text/xml',
    'text/rtf',
    'image/png',
    'image/jpeg',
    'image/webp',
    'image/heic',
    'image/heif',
    'video/mp4',
    'video/mpeg',
    'video/mov',
    'video/avi',
    'video/x-flv',
    'video/mpg',
    'video/webm',
    'video/wmv',
    'video/3gpp',
    'audio/wav',
    'audio/mp3',
    'audio/aiff',
    'audio/aac',
    'audio/ogg',
    'audio/flac'
]
# Adjust this path if your dataset is located elsewhere
load_dotenv()
# --- End Configuration ---

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gaia_evaluation_argo')

# --- GAIA Evaluation Logic ---
def extract_final_answer(response: str) -> Optional[str]:
    """Extracts the final answer from the agent's response."""
    # Look for "FINAL ANSWER: <answer>" pattern, case-insensitive, multiline
    match = re.search(r'FINAL ANSWER:\s*(.*?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback for potential variations or if the tag is missing but answer is clearly last line
    lines = response.strip().split('\n')
    if len(lines) > 1 and lines[-2].strip().upper() == "FINAL ANSWER:":
         return lines[-1].strip()
    # Check if the response itself might be the final answer if no tag is found
    # This is a heuristic, might need adjustment based on Team output format
    if len(lines) == 1:
        return lines[0].strip()
    return None

async def evaluate_gaia_benchmark(eval_dataset: Dataset, result_path: str, trace_path: str) -> None:
    """
    Evaluate the Agno Team on the GAIA benchmark dataset and save results.

    Args:
        eval_dataset: The dataset containing benchmark tasks.
        result_path: Path to save the evaluation results as JSONL.
    """
    # Create result file directory if it doesn't exist
    result_dir = os.path.dirname(result_path)
    if result_dir and not os.path.exists(result_dir):
        try:
            os.makedirs(result_dir)
            logger.info(f"Created results directory: {result_dir}")
        except OSError as e:
            logger.error(f"Error creating directory {result_dir}: {e}")
            return # Cannot proceed without results directory

    total_tasks = len(eval_dataset)
    logger.info(f"Starting GAIA evaluation on {total_tasks} tasks using Agno. Results will be saved to {result_path}")

    # --- Model Definitions (Agno) ---
    # Using placeholders where exact mapping isn't clear, adjust as needed
    model_map = {
        "claude-3-7-sonnet-latest": Claude(model="claude-3-7-sonnet-latest"),
        "gemini-2.5-flash": "gemini-2.5-flash-preview-04-17", # Use appropriate Agno ID
        "gemini-2.5-pro": "gemini-2.5-pro-preview-03-25", # Use appropriate Agno ID
        "o4-mini": ArgoLiteLLM(model="azure/o4-mini"),
        "gpt-4.1": ArgoLiteLLM(model="azure/gpt-4.1"),
    }
        
    browser_actor, browser_actor_exit_stack = await get_browser_actor(model=model_map["gemini-2.5-flash"], browser="chrome")
    browser_firefox, browser_firefox_exit_stack = await get_browser_tools(browser="firefox")
    browser_webkit, browser_webkit_exit_stack = await get_browser_tools(browser="webkit")
    
    exit_stacks = [browser_actor_exit_stack, browser_firefox_exit_stack, browser_webkit_exit_stack]

    # Define common tool lists
    common_research_tools = [
        general_search_tool,
        get_wikipedia_categories_tool,
        get_wikipedia_page_tool,
        search_wikipedia_tool,
        google_scholar_tool,
        wolfram_alpha_tool,
        get_content_from_url_tool,
        get_content_from_path_tool,
        *google_maps_tools,
        unzip_tool
    ]

    common_browser_tools = [
        general_search_tool,
        get_content_from_url_tool,
        get_content_from_path_tool,
        unzip_tool
    ]

    # --- Agent Definitions (Agno) ---
    # Defined inside the function to avoid globals

    researcher_instructions = """You have access to many tools like google search, wikipedia, google scholar, arxiv, wolfram alpha, mark it down, Youtube Understanding, yahoo finance, google maps and unzip. Please collaborate with your colleagues to accomplish the task.
In using the tool please note that you can use get_content_from_url in markitdown_tools to get the content of any web page and you can use get_content_from_path to read any local file. Please use the youtube_tools toolset to analyze any video on Youtube.
Please note that you do not have the ability to use a browser directly. If you encounter any problems that prevent you from accessing Internet resources (e.g., bot detect), please tell your browser-capable colleagues to do so.
Please note that there may be some tasks where you are unable to provide effective information, for example where you have not found any relevant or valid information, or where you are unable to read a file or document in a particular format and therefore lack relevant information. In such cases, you can simply tell your colleagues about the limitations you face and ask for their help."""
    browser_instructions = """You are able to access the Internet and interact directly with web pages through the provided browser tools. In addition to the browser tool, you can use other tools to directly use the search engine, search google, wikipedia, arxiv or Google scholar without opening the browser. These non-browser tools may return results with links to URLs that you can access using your browser. But keep in mind that the fundamental difference between you and other colleagues in terms of ability is that you have access to a browser, so make good use of it.
At any time, if you find the structure of a web page returned by a browser tool to be too complex or difficult to understand, you can also use the get_content_from_url in the markitdown_tools toolset to get more well-organized information from the page.
Your colleagues may be experiencing blocked access to some web pages, or not being able to properly fetch the content of web pages in the course of their task, so please be proactive in helping them if they point this out or request your assistance.
Please Note that You do not have the ability to watch videos or listen to audio directly from the Youtube. If you need to understand a Youtube video, please call analyze_youtube_video tool from youtube_tools toolset to analyze the video."""
    
    chat_history_prompt = "\nBefore you do any thinking or responding, read the chat history from state key 'chat_history' to understand the context of the conversation."
    
    # Researcher Alice
    alice_agent = LlmAgent(
        name="Researcher_Alice",
        description="Researcher_Alice is a researcher who can use various tools to assist in research and collaborate with colleagues.",
        model=model_map["claude-3-7-sonnet-latest"],
        instruction= "You are Researcher_Alice. " + researcher_instructions + chat_history_prompt,
        tools=common_research_tools,
        output_key="Researcher_Alice_message",
    )
    
    bob_agent = LlmAgent(
        name="Researcher_Bob",
        description="Researcher_Bob is a researcher who can use various tools to assist in research and collaborate with colleagues.",
        model=model_map["gemini-2.5-pro"],
        instruction="You are Researcher_Bob. " + researcher_instructions + chat_history_prompt,
        tools=common_research_tools,
        output_key="Researcher_Bob_message",
    )
    
    carol_agent = LlmAgent(
        name="Researcher_Carol",
        description="Researcher_Carol is a researcher who can use various tools to assist in research and collaborate with colleagues.",
        model=model_map["o4-mini"],
        instruction="You are Researcher_Carol. " + researcher_instructions + "\nYou don't need to get any permission to use tools, you can directly use any available tool whenever you think you need it." + chat_history_prompt,
        tools=common_research_tools,
        output_key="Researcher_Carol_message",
    )
    
    dave_agent = LlmAgent(
        name="Fact_Checker_Dave",
        description="Fact_Checker_Dave has access to google to assist in the task and collaborate with colleagues.",
        model=model_map["gemini-2.5-flash"],
        instruction="You are Fact_Checker_Dave. You can access the search results from Google. It is your responsibility to find information relevant to the task through Google searches and make it available to colleagues, as well as fact-checking colleagues' responses." + chat_history_prompt,
        tools=[google_search],
        output_key="Fact_Checker_Dave_message",
    )
        
    eve_agent = LlmAgent(
        name="Python_Expert_Eve",
        description="Python_Expert_Eve is a Python expert who can use Python execution tools to assist in the task and collaborate with colleagues.",
        model=model_map["gemini-2.5-flash"],
        instruction="You are Python_Expert_Eve. You have the access to python execution tools that support installing packages and executing code. You can also includes a external file or folder to your code execution." + chat_history_prompt,
        tools=[python_execution_tool],
        output_key="Python_Expert_Eve_message",
    )
    
    frank_agent = LlmAgent(
        name="Browser_Frank",
        description="Browser_Frank is a browser user who can access the Internet and interact with web pages using browser tools.",
        model=model_map["claude-3-7-sonnet-latest"],
        instruction="You are Browser_Frank. " + browser_instructions,
        tools=[*common_browser_tools, browser_actor],
        output_key="Browser_Frank_message",
    )
    
    grace_agent = LlmAgent(
        name="Browser_Grace",
        description="Browser_Grace is a browser user who can access the Internet and interact with web pages using browser tools.",
        model=model_map["gpt-4.1"],
        instruction="You are Browser_Grace. " + browser_instructions + "\nYou don't need to get any permission to use tools, you can directly use any available tool whenever you think you need it.",
        tools=[*common_browser_tools, *browser_webkit],
        output_key="Browser_Grace_message",
    )
    
    heidi_agent = LlmAgent(
        name="Browser_Heidi",
        description="Browser_Heidi is a browser user who can access the Internet and interact with web pages using browser tools.",
        model=model_map["gemini-2.5-pro"],
        instruction="You are Browser_Heidi. " + browser_instructions + "\nYou don't need to get any permission to use tools, you can directly use any available tool whenever you think you need it.",
        tools=[*common_browser_tools, *browser_firefox],
        output_key="Browser_Heidi_message",
    )
    
    colleagues = [alice_agent, bob_agent, carol_agent, dave_agent, eve_agent, frank_agent, grace_agent, heidi_agent]

    # Team Coordinator Instructions (replaces Zoe agent)
    coordinator_instructions = """You are Coordinator for a team of researchers (Alice, Bob, Carol), a fact checker (Dave), a Python expert (Eve), and browser users (Frank, Grace, Heidi).
You are responsible for coordinating collaboration and communication among all colleagues within this workgroup. 

At the end of each round of conversation you will be responsible for summarizing the progress and judge whether consensus has been reached. In order to do this, you need to use the state key 'chat_history' to get and read the chat history from previous rounds. All new messages for the current round are stored in the state with \{colleagues_name\}_message as the key, and you will also need to read all of these new messages to get a full picture of the opinions and contributions of all the colleagues too. Specifically, these keys include:
    - Researcher_Alice_message
    - Researcher_Bob_message
    - Researcher_Carol_message
    - Fact_Checker_Dave_message
    - Python_Expert_Eve_message
    - Browser_Frank_message
    - Browser_Grace_message
    - Browser_Heidi_message

If consensus has not yet been reached, you should point out the differences that still exist and encourage peers to continue the discussion in order to reach consensus. At the same time, if you find potential factual or logical errors in the peers' statements, you should kindly remind the peers to check them. If you have more specific ideas for improvement or collaboration, you may also include them.
If a consensus of final answer is reached, simply output a tag <consensus_reached/> in any part of your response, which will let the system know that you think the consensus for the final answer has been reached. Please include your final answer in your final response in the following format:

FINAL ANSWER: [YOUR FINAL ANSWER]. 

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""

    coordinator_agent = LlmAgent(
        name="Coordinator",
        description="Coordinator for a team of researchers, fact checker, and browser users.",
        model=model_map["gemini-2.5-pro"],
        instruction=coordinator_instructions,
        output_key="Coordinator_message",
    )
    
    group_chat_agent = await get_group_chat_agent(
        name="GAIA_Team",
        description="GAIA Team for collaborative research and evaluation.",
        coordinator_agent=coordinator_agent,
        group_agents=colleagues,
        max_iterations=100,
    )
    
    user_intro = f"""Hello team, we are a team of researchers, fact checkers, coders and browser users. We will be working together to accomplish many tasks. Let me now first introduce all our colleagues:
    - {"\n    - ".join([f"{agent.name}: {agent.description}" for agent in colleagues])}
    
It looks like we have a strong and well-rounded team! Let us now address the following questions together: 
"""
    
    session_service = InMemorySessionService()
    
    runner = Runner(
        app_name=APP_NAME,
        agent=group_chat_agent,
        session_service=session_service,
    )
        
    # Process each item in the dataset
    for i, item in enumerate(eval_dataset):
        # --- Task Filtering (Example: process only task 64) ---
        if i < 75:
            continue
        if i > 75:
            break
        # --- End Task Filtering ---
        
        task_id = item.get('task_id', f"task_{i}")
        question = item.get('Question', '')
        file_path = item.get('file_path', '') # Relative path

        logger.info(f"--- Processing task {task_id} ({i+1}/{total_tasks}) ---")
        
        session = session_service.create_session(app_name=APP_NAME, user_id="gaia_test_user", session_id=task_id)
        user_query = user_intro + question

        try:
            
            if file_path:
                full_file_path_obj = Path(file_path)
                full_file_path = str(full_file_path_obj)

                logger.info(f"Attempting to attach file: {full_file_path}")

                if full_file_path_obj.exists() and full_file_path_obj.is_file():
                    try:
                        user_query_with_path = user_query + f"\n\nATTACHED FILE: {full_file_path}"
                        mime_type, _ = mimetypes.guess_file_type(full_file_path)
                        if mime_type in SUPPORTED_MIMES:
                            with open(full_file_path, 'rb') as f:
                                file_content = f.read()
                                file_part = types.Part.from_bytes(data=file_content, mime_type=mime_type)
                            
                            session.state["chat_history"] = "[user]: " + user_query_with_path + "\n"
                            user_message = types.Content(role="user", parts=[file_part, types.Part(text=user_query_with_path)])
                            logger.info(f"Attaching file: {full_file_path}")
                        else:
                            attach_error = f"\n\nATTACHED FILE: [Unsupported file type: {mime_type} for file {full_file_path}]"
                            user_message = types.Content(role="user", parts=[types.Part(text=user_query+attach_error)])
                    except Exception as file_err:
                        logger.error(f"Error creating File Part for {full_file_path}: {file_err}")
                        attach_error = f"\n\nATTACHED FILE: [Error attaching file for {full_file_path}: {file_err}]"
                        user_message = types.Content(role="user", parts=[types.Part(text=user_query+attach_error)])
                else:
                    logger.warning(f"File not found or is not a file: {full_file_path}")
                    attach_error = f"\n\nATTACHED FILE: [File not found or is not a file: {full_file_path}]"
                    user_message = types.Content(role="user", parts=[types.Part(text=user_query+attach_error)])
            else:
                user_message = types.Content(role="user", parts=[types.Part(text=user_query)])
                logger.info("No file path provided for this task.")

            logger.info("Sending task to team...")
            
            # Run the team
            async for event in runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=user_message,
            ):
                # Check for final response
                if event.is_final_response() and event.content:
                    # logger.info(f"Final response received for task {task_id}.")
                    final_response = event.content
                    # print for debugging:
                    print(f"\n\nFinal response content: {final_response}\n\n")
                

            final_response_text = "\n".join([part.text for part in final_response.parts if part.text])

            # Extract final answer
            final_answer = extract_final_answer(final_response_text)
            if not final_answer:
                logger.warning(f"No 'FINAL ANSWER:' found in response for task {task_id}. Using full response as reasoning trace.")
                final_answer = "NO_FINAL_ANSWER_FOUND" # Mark as not found
            else:
                logger.info(f"Extracted Final Answer: {final_answer}")

            # Prepare result dictionary
            result = {
                "task_id": task_id,
                "model_answer": final_answer,
                "reasoning_trace": final_response_text # Save the full response text
            }

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {str(e)}", exc_info=True)
            # Log the error in the results file
            result = {
                "task_id": task_id,
                "error": str(e),
                "status": "failed",
                "reasoning_trace": None # No response trace available
            }

        # Append result to the JSONL file
        try:
            with open(result_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
            logger.info(f"Result for task {task_id} saved.")
        except IOError as e:
                logger.error(f"Error writing result for task {task_id} to {result_path}: {e}")

        try:
            with open(trace_path, 'a') as f:
                f.write(f"\n---[Task ID: {task_id}]---\n" + session.state["chat_history"] + '\n---END OF TASK---\n')
            logger.info(f"Trace for task {task_id} saved.")
        except IOError as e:
            logger.error(f"Error writing trace for task {task_id} to {trace_path}: {e}")
        
        logger.info(f"--- Finished task {task_id} ---")

    logger.info(f"GAIA evaluation completed. Results saved to {result_path}")
    # for stack in exit_stacks:
    #     await stack.aclose()
    # logger.info("All MCP Tools closed.")

# --- Main Execution Logic ---
async def main():
    """Load GAIA dataset and run the evaluation."""
    try:
        logger.info("Loading GAIA dataset...")
        # Load only the validation split as needed
        gaia_validation_dataset = datasets.load_dataset("gaia-benchmark/GAIA", '2023_all', split="validation")
        logger.info("Dataset loaded.")

        logger.info("Starting evaluation...")
        await evaluate_gaia_benchmark(gaia_validation_dataset, RESULT_FILE_VALIDATION, TRACE_FILE_VALIDATION)

        logger.info("Evaluation finished.")

    except Exception as e:
        logger.critical(f"An critical error occurred during the main execution: {e}", exc_info=True)

if __name__ == "__main__":
    # Ensure the script runs the asyncio event loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user.")
    except Exception as e:
        logger.critical(f"Script failed: {e}", exc_info=True)
