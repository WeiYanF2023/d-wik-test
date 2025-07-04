# Modified from google.adk.models.anthropic_llm

"""Anthropic integration for Claude models."""

from __future__ import annotations

from functools import cached_property
import logging
import os
import base64
from typing import Any, AsyncGenerator
from typing import Iterable
from typing import Literal
from typing import Optional, Union
from typing import TYPE_CHECKING

from anthropic import AsyncAnthropic, AsyncStream
from anthropic.resources.beta import AsyncBeta
from anthropic import NOT_GIVEN
from anthropic.types import beta as anthropic_beta_types
from anthropic import types as anthropic_types
from google.genai import types
from pydantic import BaseModel
from typing_extensions import override

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse

if TYPE_CHECKING:
    from google.adk.models.llm_request import LlmRequest

__all__ = ["Claude"]

logger = logging.getLogger(__name__)

MAX_TOKEN = 64000


class ClaudeRequest(BaseModel):
    system_instruction: str
    messages: Iterable[anthropic_beta_types.BetaMessageParam]
    tools: list[anthropic_beta_types.BetaToolParam]


def to_claude_role(role: Optional[str]) -> Literal["user", "assistant"]:
    if role in ["model", "assistant"]:
        return "assistant"
    return "user"


def to_google_genai_finish_reason(
    anthropic_stop_reason: Optional[str],
) -> types.FinishReason:
    if anthropic_stop_reason in ["end_turn", "stop_sequence", "tool_use"]:
        return "STOP"
    if anthropic_stop_reason == "max_tokens":
        return "MAX_TOKENS"
    return "FINISH_REASON_UNSPECIFIED"


def part_to_message_block(
    part: types.Part,
) -> Union[
    anthropic_beta_types.BetaTextBlockParam,
    anthropic_beta_types.BetaImageBlockParam,
    anthropic_beta_types.BetaToolUseBlockParam,
    anthropic_beta_types.BetaToolResultBlockParam,
    anthropic_types.DocumentBlockParam,
]:
    if part.text:
        return anthropic_beta_types.BetaTextBlockParam(text=part.text, type="text")
    if part.inline_data:
        if part.inline_data.mime_type.startswith("image"):
            return anthropic_beta_types.BetaImageBlockParam(
                source=anthropic_beta_types.BetaBase64ImageSourceParam(
                    data=base64.b64encode(part.inline_data.data).decode("utf-8"),
                    mime_type=part.inline_data.mime_type,
                    type="base64",
                ),
                type="image",
            )
        if (
            part.inline_data.mime_type == "application/pdf"
            or part.inline_data.mime_type == "application/x-pdf"
        ):
            return anthropic_types.DocumentBlockParam(
                source=anthropic_beta_types.BetaBase64PDFSourceParam(
                    data=base64.b64encode(part.inline_data.data).decode("utf-8"),
                    media_type="application/pdf",
                    type="base64",
                ),
                type="document",
            )
    if part.function_call:
        assert part.function_call.name

        return anthropic_beta_types.BetaToolUseBlockParam(
            id=part.function_call.id or "",
            name=part.function_call.name,
            input=part.function_call.args,
            type="tool_use",
        )
    if part.function_response:
        content = ""
        if (
            "result" in part.function_response.response
            and part.function_response.response["result"]
        ):
            # Transformation is required because the content is a list of dict.
            # ToolResultBlockParam content doesn't support list of dict. Converting
            # to str to prevent anthropic.BadRequestError from being thrown.
            content = str(part.function_response.response["result"])
        return anthropic_beta_types.BetaToolResultBlockParam(
            tool_use_id=part.function_response.id or "",
            type="tool_result",
            content=content,
            is_error=False,
        )
    return anthropic_beta_types.BetaTextBlockParam(
        text="[Unsupported Content Type]",
        type="text",
    )


def content_to_message_param(
    content: types.Content,
) -> anthropic_beta_types.BetaMessageParam:
    return {
        "role": to_claude_role(content.role),
        "content": [part_to_message_block(part) for part in content.parts or []],
    }


def content_block_to_part(
    content_block: anthropic_beta_types.BetaContentBlock,
) -> types.Part:
    if isinstance(content_block, anthropic_beta_types.BetaTextBlock):
        return types.Part.from_text(text=content_block.text)
    if isinstance(content_block, anthropic_beta_types.BetaToolUseBlock):
        assert isinstance(content_block.input, dict)
        part = types.Part.from_function_call(
            name=content_block.name, args=content_block.input
        )
        part.function_call.id = content_block.id
        return part
    return types.Part.from_text(
        text="[Unsupported Content Type]",
    )


def message_to_generate_content_response(
    message: anthropic_beta_types.BetaMessage,
) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[content_block_to_part(cb) for cb in message.content],
        ),
        # TODO: Deal with these later.
        # finish_reason=to_google_genai_finish_reason(message.stop_reason),
        # usage_metadata=types.GenerateContentResponseUsageMetadata(
        #     prompt_token_count=message.usage.input_tokens,
        #     candidates_token_count=message.usage.output_tokens,
        #     total_token_count=(
        #         message.usage.input_tokens + message.usage.output_tokens
        #     ),
        # ),
    )

def _type_to_string(value_dict: dict[str, Any]|list):
    if isinstance(value_dict, list):
        for item in value_dict:
            _type_to_string(item)
    if isinstance(value_dict, dict):
        if "type" in value_dict:
            value_dict["type"] = value_dict["type"].lower()
        for key, value in value_dict.items():
            if isinstance(value, dict):
                _type_to_string(value)
            elif isinstance(value, list):
                for item in value:
                    _type_to_string(item)
            

def function_declaration_to_tool_param(
    function_declaration: types.FunctionDeclaration,
) -> anthropic_beta_types.BetaToolParam:
    assert function_declaration.name

    properties = {}
    if function_declaration.parameters and function_declaration.parameters.properties:
        for key, value in function_declaration.parameters.properties.items():
            value_dict = value.model_dump(exclude_none=True)
            _type_to_string(value_dict)
            properties[key] = value_dict

    return anthropic_beta_types.BetaToolParam(
        name=function_declaration.name,
        description=function_declaration.description or "",
        input_schema={
            "type": "object",
            "properties": properties,
        },
    )


class Claude(BaseLlm):
    model: str = "claude-3-7-sonnet-20250219"

    @staticmethod
    @override
    def supported_models() -> list[str]:
        return [r"claude-3-.*"]

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        messages = [
            content_to_message_param(content) for content in llm_request.contents or []
        ]
        tools = NOT_GIVEN
        if (
            llm_request.config
            and llm_request.config.tools
            and llm_request.config.tools[0].function_declarations
        ):
            tools = [
                function_declaration_to_tool_param(tool)
                for tool in llm_request.config.tools[0].function_declarations
            ]
        tool_choice = (
            anthropic_beta_types.BetaToolChoiceAutoParam(
                type="auto",
                # TODO: allow parallel tool use.
                disable_parallel_tool_use=True,
            )
            if llm_request.tools_dict
            else NOT_GIVEN
        )
        message = await self._anthropic_client.messages.create(
            model=llm_request.model,
            system=llm_request.config.system_instruction,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=MAX_TOKEN,
            betas=["token-efficient-tools-2025-02-19"],
            timeout=3600.0,
        )
        logger.info(
            "Claude response: %s",
            message.model_dump_json(indent=2, exclude_none=True),
        )
        yield message_to_generate_content_response(message)

    @cached_property
    def _anthropic_client(self) -> AsyncBeta:
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set for using Anthropic Models."
            )

        return AsyncAnthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
        ).beta
