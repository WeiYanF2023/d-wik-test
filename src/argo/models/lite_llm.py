import base64
import logging
from typing import AsyncGenerator, Iterable, Union
from litellm import (
    ChatCompletionAssistantMessage,
    ChatCompletionAudioObject,
    ChatCompletionDeveloperMessage,
    ChatCompletionImageUrlObject,
    ChatCompletionMessageToolCall,
    ChatCompletionTextObject,
    ChatCompletionToolMessage,
    ChatCompletionUserMessage,
    ChatCompletionVideoUrlObject,
    ChatCompletionDocumentObject,
    DocumentObject,
    Function,
    Message,
    OpenAIMessageContent,
)
from google.adk.models.lite_llm import (
    LiteLlm,
    LlmRequest,
    LlmResponse,
    FunctionChunk,
    TextChunk,
    _safe_json_serialize,
    _to_litellm_role,
    _function_declaration_to_tool_param,
    _model_response_to_chunk,
    _build_request_log,
    _message_to_generate_content_response,
    _model_response_to_generate_content_response
)
from google.genai import types
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio

logger = logging.getLogger(__name__)

class ArgoLiteLLM(LiteLlm):
    """ArgoLiteLLM is a wrapper around LiteLlm to provide
    additional functionality specific to the Argo framework.
    """

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Generates content asynchronously.

        Args:
            llm_request: LlmRequest, the request to send to the LiteLlm model.
            stream: bool = False, whether to do streaming call.

        Yields:
            LlmResponse: The model response.
        """

        logger.info(_build_request_log(llm_request))

        messages, tools = _get_completion_inputs(llm_request)

        completion_args = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
        }
        completion_args.update(self._additional_args)

        if stream:
            text = ""
            function_name = ""
            function_args = ""
            function_id = None
            completion_args["stream"] = True
            for part in self.llm_client.completion(**completion_args):
                for chunk, finish_reason in _model_response_to_chunk(part):
                    if isinstance(chunk, FunctionChunk):
                        if chunk.name:
                            function_name += chunk.name
                        if chunk.args:
                            function_args += chunk.args
                        function_id = chunk.id or function_id
                    elif isinstance(chunk, TextChunk):
                        text += chunk.text
                        yield _message_to_generate_content_response(
                            ChatCompletionAssistantMessage(
                                role="assistant",
                                content=chunk.text,
                            ),
                            is_partial=True,
                        )
                    if finish_reason == "tool_calls" and function_id:
                        yield _message_to_generate_content_response(
                            ChatCompletionAssistantMessage(
                                role="assistant",
                                content="",
                                tool_calls=[
                                    ChatCompletionMessageToolCall(
                                        type="function",
                                        id=function_id,
                                        function=Function(
                                            name=function_name,
                                            arguments=function_args,
                                        ),
                                    )
                                ],
                            )
                        )
                        function_name = ""
                        function_args = ""
                        function_id = None
                    elif finish_reason == "stop" and text:
                        yield _message_to_generate_content_response(
                            ChatCompletionAssistantMessage(role="assistant", content=text)
                        )
                        text = ""

        else:
            response = await self.llm_client.acompletion(**completion_args)
            yield _model_response_to_generate_content_response(response)


def _get_completion_inputs(
    llm_request: LlmRequest,
) -> tuple[Iterable[Message], Iterable[dict]]:
    """Converts an LlmRequest to litellm inputs.

    Args:
      llm_request: The LlmRequest to convert.

    Returns:
      The litellm inputs (message list and tool dictionary).
    """
    messages = []
    for content in llm_request.contents or []:
        message_param_or_list = _content_to_message_param(content)
        if isinstance(message_param_or_list, list):
            messages.extend(message_param_or_list)
        elif message_param_or_list:  # Ensure it's not None before appending
            messages.append(message_param_or_list)

    if llm_request.config.system_instruction:
        messages.insert(
            0,
            ChatCompletionDeveloperMessage(
                role="developer",
                content=llm_request.config.system_instruction,
            ),
        )

    tools = None
    if (
        llm_request.config
        and llm_request.config.tools
        and llm_request.config.tools[0].function_declarations
    ):
        tools = [
            _function_declaration_to_tool_param(tool)
            for tool in llm_request.config.tools[0].function_declarations
        ]
    return messages, tools


def _content_to_message_param(
    content: types.Content,
) -> Union[Message, list[Message]]:
    """Converts a types.Content to a litellm Message or list of Messages.

    Handles multipart function responses by returning a list of
    ChatCompletionToolMessage objects if multiple function_response parts exist.

    Args:
      content: The content to convert.

    Returns:
      A litellm Message, a list of litellm Messages.
    """

    tool_messages = []
    for part in content.parts:
        if part.function_response:
            tool_messages.append(
                ChatCompletionToolMessage(
                    role="tool",
                    tool_call_id=part.function_response.id,
                    content=_safe_json_serialize(part.function_response.response),
                )
            )
    if tool_messages:
        return tool_messages if len(tool_messages) > 1 else tool_messages[0]

    # Handle user or assistant messages
    role = _to_litellm_role(content.role)
    message_content = _get_content(content.parts) or None

    if role == "user":
        return ChatCompletionUserMessage(role="user", content=message_content)
    else:  # assistant/model
        tool_calls = []
        content_present = False
        for part in content.parts:
            if part.function_call:
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        type="function",
                        id=part.function_call.id,
                        function=Function(
                            name=part.function_call.name,
                            arguments=part.function_call.args,
                        ),
                    )
                )
            elif part.text or part.inline_data:
                content_present = True

        final_content = message_content if content_present else None

        return ChatCompletionAssistantMessage(
            role=role,
            content=final_content,
            tool_calls=tool_calls or None,
        )


def _get_content(
    parts: Iterable[types.Part],
) -> Union[OpenAIMessageContent, str]:
    """Converts a list of parts to litellm content.

    Args:
    parts: The parts to convert.

    Returns:
    The litellm content.
    """

    content_objects = []
    for part in parts:
        if part.text:
            if len(parts) == 1:
                return part.text
            content_objects.append(
                ChatCompletionTextObject(
                    type="text",
                    text=part.text,
                )
            )
        elif part.inline_data and part.inline_data.data and part.inline_data.mime_type:
            base64_string = base64.b64encode(part.inline_data.data).decode("utf-8")
            data_uri = f"data:{part.inline_data.mime_type};base64,{base64_string}"

            if part.inline_data.mime_type.startswith("image"):
                content_objects.append(
                    ChatCompletionImageUrlObject(
                        type="image_url",
                        image_url=data_uri,
                    )
                )
            # TODO: Add support for other files
            # elif part.inline_data.mime_type.startswith("video"):
            #     content_objects.append(
            #         ChatCompletionVideoUrlObject(
            #             type="video_url",
            #             video_url=data_uri,
            #         )
            #     )
            # elif part.inline_data.mime_type.startswith("audio"):
            #     audio_format = part.inline_data.mime_type.split("/")[1]
            #     if audio_format == "mpeg":
            #         audio_format = "mp3"
            #     content_objects.append(
            #         ChatCompletionAudioObject(
            #             type="input_audio",
            #             input_audio=InputAudio(
            #                 data=base64_string,
            #                 format=audio_format,
            #             ),
            #         )
            #     )
            elif part.inline_data.mime_type.startswith("text"):
                content_objects.append(
                    ChatCompletionTextObject(
                        type="text",
                        text="[Content of File]\n"
                        + part.inline_data.data.decode("utf-8"),
                    )
                )
            # elif part.inline_data.mime_type in [
            #     "application/pdf",
            #     "application/x-pdf",
            # ]:
            #     content_objects.append(
            #         ChatCompletionDocumentObject(
            #             type="document",
            #             source=DocumentObject(
            #                 type="text",
            #                 data=base64_string,
            #                 media_type=part.inline_data.mime_type,
            #             ),
            #         )
            #     )
            else:
                content_objects.append(
                    ChatCompletionTextObject(
                        type="text",
                        text="\n[Unsupported Content]\n",
                    )
                )
                # raise ValueError("LiteLlm(BaseLlm) does not support this content part.")

    return content_objects
