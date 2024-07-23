import os
from typing import Any, Coroutine, List, Literal, Optional, Union, overload

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai_messages_token_helper import build_messages, get_token_limit

from approaches.approach import ThoughtStep
from approaches.chatapproach import ChatApproach
from core.authentication import AuthenticationHelper

import logging

import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings, OpenAIChatPromptExecutionSettings
)
#from services import Service
#from samples.service_settings import ServiceSettings
from semantic_kernel.planners.function_calling_stepwise_planner import (
    FunctionCallingStepwisePlanner,
    FunctionCallingStepwisePlannerOptions,
)
from plugins.api_plugin import ApiPlugin


class ChatReadRetrieveReadApproach(ChatApproach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then uses Azure AI Search to retrieve relevant documents, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_model: str,
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    @property
    def system_message_chat_conversation(self):
        return """Assistant helps the company employees with their healthcare plan questions, and questions about the employee handbook. Be brief in your answers.
        Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.
        For tabular information return it as an html table. Do not return markdown format. If the question is not in English, answer in the language used in the question.
        Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brackets to reference the source, for example [info1.txt]. Don't combine sources, list each source separately, for example [info1.txt][info2.pdf].
        {follow_up_questions_prompt}
        {injected_prompt}
        """

    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[False],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, ChatCompletion]]: ...

    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[True],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, AsyncStream[ChatCompletionChunk]]]: ...
    
    
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]]]:
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        filter = self.build_filter(overrides, auth_claims)
            
        original_user_query = messages[-1]["content"]
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")
        user_query_request = "Generate search query for: " + original_user_query

        tools: List[ChatCompletionToolParam] = [
            {
                "type": "function",
                "function": {
                    "name": "search_sources",
                    "description": "Retrieve sources from the Azure AI Search index",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "Query string to retrieve documents from azure search eg: 'Health care plan'",
                            }
                        },
                        "required": ["search_query"],
                    },
                },
            }
        ]

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        query_response_token_limit = 100
        query_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=self.query_prompt_template,
            tools=tools,
            few_shots=self.query_prompt_few_shots,
            past_messages=messages[:-1],
            new_user_content=user_query_request,
            max_tokens=self.chatgpt_token_limit - query_response_token_limit,
        )

        # print("query_MSG_started")
        # print(query_messages)
        # print("query_MSG_ended")
        
        chat_completion: ChatCompletion = await self.openai_client.chat.completions.create(
            messages=query_messages,  # type: ignore
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            temperature=0.0,  # Minimize creativity for search query generation
            max_tokens=query_response_token_limit,  # Setting too low risks malformed JSON, setting too high may affect performance
            n=1,
            tools=tools,
        )
        #print(chat_completion.choices[0])
        query_text = self.get_search_query(chat_completion, original_user_query)
        
        # print("query_text_started")
        # print(query_text)
        # print("query_text_ended")
        # Initialize the kernel
        kernel = Kernel()
        #Add Azure OpenAI chat completion
        # kernel.add_service(await self.openai_client.chat.completions.create(
        #     messages=query_messages,  # type: ignore
        #     # Azure OpenAI takes the deployment name as the model name
        #     model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
        #     temperature=0.0,  # Minimize creativity for search query generation
        #     max_tokens=query_response_token_limit,  # Setting too low risks malformed JSON, setting too high may affect performance
        #     n=1,
        #     tools=tools
        # ))
        service_id = "default_1"
        kernel.add_service(
            AzureChatCompletion(
                service_id=service_id,
                api_version="2023-05-15",  # Use the appropriate version
                deployment_name="sematic-kernel-test",
                api_key="ce2cdbcb860b41ed95723ae3aba17bc7",
                endpoint="https://cog-qpy3j27u2zsoo.openai.azure.com/"
            ),
        )
        logging.basicConfig(
            format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.getLogger("kernel").setLevel(logging.DEBUG)

        plugins_directory = "prompt_template_samples"
        
        api_plugin = kernel.add_plugin(plugin=ApiPlugin(), plugin_name="api")
        kybplugin_function = kernel.add_plugin(parent_directory=plugins_directory, plugin_name="KYBData")
        APIDataFunction = api_plugin["FetchBusinessData"]
        KYBReportFunction = kybplugin_function["kyb_data"]
        
        execution_settings = AzureChatPromptExecutionSettings(
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            temperature=0.0,  # Minimize creativity for search query generation
            max_tokens=query_response_token_limit,  # Setting too low risks malformed JSON, setting too high may affect performance
            n=1,
            tools=tools,
            tool_choice="auto",
            function_call_behavior=FunctionCallBehavior.EnableFunctions(auto_invoke=True, filters={"included_plugins": ["api","KYBData"]})
            )
        
        #execution_settings.function_call_behavior = FunctionCallBehavior.EnableFunctions(auto_invoke=True, filters={"included_plugins": ["Testing"]})
        
        # test_func = kernel.add_function(
        #     prompt = "{{$chat_history}}{{$user_input}}",
        #     plugin_name="Lights",
        #     function_name="get_lights",
        # )
        
        # # Create a planner
        # planner = ActionPlanner(kernel)

        # # Define the goal for the planner
        # async def process_user_input(user_input, chat_history):
        #     goal = f"""
        #     Analyze the following user input and chat history, then determine the most appropriate action:
        #     User Input: {user_input}
        #     Chat History: {chat_history}
            
        #     If the context is related to KYB (Know Your Business) or KYC (Know Your Customer), use the FetchBusinessData function.
        #     If the context is related to generating a KYB report, use the kyb_data function.
        #     If the context is not related to KYB or KYC, respond with "No relevant function available".
            
        #     Respond with the name of the function to use or the message "No relevant function available".
        #     """
            
        #     # Get the plan
        #     plan = await planner.create_plan(goal)
            
        #     # Execute the plan
        #     result = await planner.execute_plan(plan)
            
        #     return result

            # List of questions to process
            
        chat = ChatHistory()
        question = f"""Analyze the following user input which contains both current query and chat history, then determine the most appropriate action:
            User Input: {query_text}
            
            if you decide to call "FetchBusinessData" dont worry about parameters or API key. They are already provided in function code. Just call function and return response and store it in result.final_answer.

            If the context is related to KYB (Know Your Business) or KYC (Know Your Customer), Fetch data from FetchBusinessData and create a report for that data using KYBData.
            If the context is related to generating a KYB report, Fetch data from FetchBusinessData and create a report for that data using KYBData.
            If the context is not related to KYB or KYC, respond with "No relevant function available".

            Respond with only one of the above options.""",
        
    
        # Configure planner options
        options = FunctionCallingStepwisePlannerOptions(
            max_iterations=10,
            max_tokens=4000,
        )
    
        # Initialize the planner
        planner = FunctionCallingStepwisePlanner(service_id=service_id, options=options)

        try:
            result = await planner.invoke(kernel, question)
            # print(f"Q: {question}\nA: {result}\n")
            # print()
            # print()
            print(f"Answer:    {result.final_answer}")
            #print(result.final_answer)
            # print(result.chat_history)
            # print("Checking Format")
            # print()
            # print()
        except Exception as e:
            logging.error(f"Error processing question '{question}': {e}")
            
        arguments = KernelArguments(settings=execution_settings)
        # print("Response")
        # print(chat_completion.choices[0])
        # print("Responses_1")
        #arguments["user_input"] = chat_completion.choices[0].message.content
        #print("User Input")
        #print(query_messages)
        arguments["user_input"] = query_messages
        arguments["chat_history"] = chat
        #print("Result")
        #result = await kernel.invoke(test_func, arguments= arguments)
        # response_1 = await kernel.invoke(APIDataFunction, arguments= arguments)
        # # Process the user input
        # #result = await process_user_input(arguments["user_input"], arguments["chat_history"])
        # result_1 = await kernel.invoke(KYBReportFunction, input=response_1)
        # print("Response_1")
        # print(response_1)
        # print(result_1)
        # print("Result_1")
        # #print(result)
        # #print(result)
        # #print(result)
        # print("New code")
        
        # print("sematic start")
        # chat_completion_semantic : AzureChatCompletion = kernel.get_service(type=ChatCompletionClientBase)
        
        #content = "You are an AI assistant that helps people find information."
        #chat.add_system_message(content)
        #chat.add_user_message("Gets a list of lights and their current state")
        #stream_1 = chat_completion_semantic.get_chat_message_contents(chat_history=chat , settings = execution_settings)
        #print(stream_1)
        #print("Stream 1 ends here")
        # stream = chat_completion_semantic.get_streaming_chat_message_contents(
        #     chat_history=chat, settings = execution_settings
        # )
        # print(stream)
        # async for text in stream:
        #     print(str(text[0]), end="")  # end = "" to avoid newlines
        
        print("semantic end")
        #query_text = self.get_search_query(chat_completion, original_user_query)
        
        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if use_vector_search:
            vectors.append(await self.compute_text_embedding(query_text))

        results = await self.search(
            top,
            query_text,
            filter,
            vectors,
            use_text_search,
            use_vector_search,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
        )

        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)
        content = "\n".join(sources_content)

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        system_message = self.get_system_prompt(
            overrides.get("prompt_template"),
            self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else "",
        )

        response_token_limit = 1024
        messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=system_message,
            past_messages=messages[:-1],
            # Model does not handle lengthy system messages well. Moving sources to latest user conversation to solve follow up questions prompt.
            new_user_content=original_user_query + "\n\nSources:\n" + content,
            max_tokens=self.chatgpt_token_limit - response_token_limit,
        )

        data_points = {"text": sources_content}

        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Prompt to generate search query",
                    [str(message) for message in query_messages],
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
                ThoughtStep(
                    "Search using generated search query",
                    query_text,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "top": top,
                        "filter": filter,
                        "use_vector_search": use_vector_search,
                        "use_text_search": use_text_search,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
                ThoughtStep(
                    "Prompt to generate answer",
                    [str(message) for message in messages],
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
            ],
        }

        print("coroutie")
        chat_coroutine = self.openai_client.chat.completions.create(
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            messages=messages,
            temperature=overrides.get("temperature", 0.3),
            max_tokens=response_token_limit,
            n=1,
            stream=should_stream,
        )
        
        print("start")
        print(chat_coroutine)
        print("stop")
         
        # Get the response from the AI
        #result = (await chat_completion.get_chat_message_contents(
         #   chat_history=history,
          #  settings=execution_settings,
           # kernel=kernel,
            #,
        #))[0]###
        return (extra_info, chat_coroutine)
