"""Group Chat Workflow Agent Implementation."""

import re
from typing import AsyncGenerator, List

from google.adk.agents import (
    BaseAgent,
    SequentialAgent,
    ParallelAgent,
    LoopAgent,
)
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from argo.utils import check_tag_presence


class _CheckConsensusAndEscalate(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        yield Event(
            author=self.name if self.name else "Consensus Check",
            actions=EventActions(
                escalate=check_tag_presence(
                    ctx.session.state.get("Coordinator_message", ""),
                    "consensus_reached",
                )
            ),
        )


class _UpdateChatHistory(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Update the chat history with the latest message.
        name_pattern = re.compile(r"^(?P<name>.+)_message$")
        new_messages = {
            name_pattern.match(k).group("name"): v
            for k, v in ctx.session.state.items()
            if name_pattern.match(k)
        }
        updated_chat_history: str = ctx.session.state.get("chat_history", "") + "\n"
        updated_chat_history += "\n".join(
            f"[{name}]: {message}\n" for name, message in new_messages.items()
        )
        print("!!! DEBUG: Updated chat history:", updated_chat_history)
        yield Event(
            author=self.name,
            actions=EventActions(
                state_delta={"chat_history": updated_chat_history},
            ),
        )


async def get_group_chat_agent(
    name: str,
    description: str,
    coordinator_agent: BaseAgent,
    group_agents: List[BaseAgent],
    max_iterations: int = 10,
) -> LoopAgent:
    """Creates a group chat agent that coordinates multiple agents.

    Args:
        name: The name of the group chat agent.
        description: The description of the group chat agent.
        coordinator_agent: The coordinator agent that manages the group chat.
        group_agents: A list of agents that will participate in the group chat.
        max_iterations: The maximum number of rounds for the group chat.

    Returns:
        A LoopAgent instance representing the group chat agent.
    """
    parallel_agents = ParallelAgent(
        name="Working_Group",
        description="Parallel working group agents",
        sub_agents=group_agents,
    )
    update_chat_history = _UpdateChatHistory(name="Update_Chat_History")
    check_consensus = _CheckConsensusAndEscalate(name="Check_Consensus")

    return LoopAgent(
        name=name,
        description=description,
        sub_agents=[
            parallel_agents,
            coordinator_agent,
            update_chat_history,
            check_consensus,
        ],
        max_iterations=max_iterations,
    )