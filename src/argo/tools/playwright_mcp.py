from contextlib import AsyncExitStack
from typing import List, Tuple
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams, StdioServerParameters, MCPTool

async def get_browser_tools(browser: str = "firefox") -> Tuple[List[MCPTool], AsyncExitStack]:

  tools, exit_stack = await MCPToolset.from_server(
      connection_params=StdioServerParameters(
          command='npx',
          args=["-y",
                "@playwright/mcp@latest",
                "--browser", 
                browser,
          ]
      )
  )
  print("MCP Toolset created successfully.")
  return tools, exit_stack