import json
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from approaches.approach import Approach


class ChatApproach(Approach, ABC):
    # Chat roles
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    query_prompt_few_shots = [
        {"role": USER, "content": "What are key areas of the Bank Secrecy Act?"},
        {"role": ASSISTANT, "content": "Summarize the Bank Secrecy Act and reference the actual regulation, before referencing handbooks or manuals. Search query: Bank Secrecy Act regulation key areas source:regulation"},
        {"role": USER, "content": "What is required for a Community Reinvestment Act audit?"},
        {"role": ASSISTANT, "content": "Show the main topics covered in a Community Reinvestment Act audit and prioritize manuals and handbooks as primary sources. Search query: Community Reinvestment Act audit requirements source:handbook"},
        {"role": USER, "content": "Generate search query for: Can you explain the differences between US banking Regulation O and Regulation W?"},
        {"role": ASSISTANT, "content": "Explain the main differences between US banking Regulation O and Regulation W. Prioritize using regulations to answer and only use manuals and handbooks as secondary sources. Search query: differences between US banking Regulation O Regulation W source:regulation"},
        {"role": USER, "content": "What are the key provisions of the Truth in Lending Act (TILA)?"},
        {"role": ASSISTANT, "content": "Show the main topics within the Truth in Lending Act (TILA). Prioritize using regulations in your answer and only use manuals and handbooks as secondary sources. Search query: Truth in Lending Act TILA key provisions source:regulation"},
    ]
    NO_RESPONSE = "0"

    follow_up_questions_prompt_content = """Generate 3 very brief follow-up questions that the user would likely ask next.
    Enclose the follow-up questions in double angle brackets. Example:
    <<Who is not covered by EU Privacy Regulations?>>
    <<What are the responsibilties of the customer during credit desputes?>>
    <<If customer short pays, can interest be charged by creditor on balance?>>
    Do no repeat questions that have already been asked.
    Make sure the last question ends with ">>".
    """

    query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be 
    answered by searching in a knowledge base.
    You have access to Azure AI Search index with 100's of documents, including regulations and handbooks.
    Generate a search query based on the conversation and the new question.
    Determine whether to prioritize regulations or handbooks based on the nature of the question.
    Add 'source:regulation' or 'source:handbook' to the end of the query to indicate the preferred source.
    Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
    Do not include any text inside [] or <<>> in the search query terms.
    Do not include any special characters like '+'.
    If the question is not in English, translate the question to English before generating the search query.
    If you cannot generate a search query, return just the number 0.
    """

    @property
    @abstractmethod
    def system_message_chat_conversation(self) -> str:
        pass

    @abstractmethod
    async def run_until_final_call(self, messages, overrides, auth_claims, should_stream) -> tuple:
        pass

    def get_system_prompt(self, override_prompt: Optional[str], follow_up_questions_prompt: str) -> str:
        if override_prompt is None:
            return self.system_message_chat_conversation.format(
                injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt
            )
        elif override_prompt.startswith(">>>"):
            return self.system_message_chat_conversation.format(
                injected_prompt=override_prompt[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt
            )
        else:
            return override_prompt.format(follow_up_questions_prompt=follow_up_questions_prompt)

    def get_search_query(self, chat_completion: ChatCompletion, user_query: str):
        response_message = chat_completion.choices[0].message

        if response_message.tool_calls:
            for tool in response_message.tool_calls:
                if tool.type != "function":
                    continue
                function = tool.function
                if function.name == "search_sources":
                    arg = json.loads(function.arguments)
                    search_query = arg.get("search_query", self.NO_RESPONSE)
                    if search_query != self.NO_RESPONSE:
                        return search_query
        elif query_text := response_message.content:
            if query_text.strip() != self.NO_RESPONSE:
                # Check if source is specified in the query
                if "source:regulation" in query_text or "source:handbook" in query_text:
                    return query_text
                else:
                    # If no source is specified, add a default
                    return f"{query_text} source:regulation"
        return f"{user_query} source:regulation"  # Default to regulation if no query is generated

    def extract_followup_questions(self, content: str):
        return content.split("<<")[0], re.findall(r"<<([^>>]+)>>", content)

    async def run_without_streaming(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> dict[str, Any]:
        extra_info, chat_coroutine = await self.run_until_final_call(
            messages, overrides, auth_claims, should_stream=False
        )
        chat_completion_response: ChatCompletion = await chat_coroutine
        chat_resp = chat_completion_response.model_dump()  # Convert to dict to make it JSON serializable
        chat_resp = chat_resp["choices"][0]
        chat_resp["context"] = extra_info
        if overrides.get("suggest_followup_questions"):
            content, followup_questions = self.extract_followup_questions(chat_resp["message"]["content"])
            chat_resp["message"]["content"] = content
            chat_resp["context"]["followup_questions"] = followup_questions
        chat_resp["session_state"] = session_state
        return chat_resp

    async def run_with_streaming(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> AsyncGenerator[dict, None]:
        extra_info, chat_coroutine = await self.run_until_final_call(
            messages, overrides, auth_claims, should_stream=True
        )
        yield {"delta": {"role": "assistant"}, "context": extra_info, "session_state": session_state}

        followup_questions_started = False
        followup_content = ""
        async for event_chunk in await chat_coroutine:
            # "2023-07-01-preview" API version has a bug where first response has empty choices
            event = event_chunk.model_dump()  # Convert pydantic model to dict
            if event["choices"]:
                completion = {"delta": event["choices"][0]["delta"]}
                # if event contains << and not >>, it is start of follow-up question, truncate
                content = completion["delta"].get("content")
                content = content or ""  # content may either not exist in delta, or explicitly be None
                if overrides.get("suggest_followup_questions") and "<<" in content:
                    followup_questions_started = True
                    earlier_content = content[: content.index("<<")]
                    if earlier_content:
                        completion["delta"]["content"] = earlier_content
                        yield completion
                    followup_content += content[content.index("<<") :]
                elif followup_questions_started:
                    followup_content += content
                else:
                    yield completion
        if followup_content:
            _, followup_questions = self.extract_followup_questions(followup_content)
            yield {"delta": {"role": "assistant"}, "context": {"followup_questions": followup_questions}}

    async def run(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        return await self.run_without_streaming(messages, overrides, auth_claims, session_state)

    async def run_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> AsyncGenerator[dict[str, Any], None]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        return self.run_with_streaming(messages, overrides, auth_claims, session_state)
