import asyncio
import json
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from dotenv import load_dotenv
import logging
from semantic_kernel.planners.function_calling_stepwise_planner import (
    FunctionCallingStepwisePlanner,
    FunctionCallingStepwisePlannerOptions,
)
from api_plugin import ApiPlugin
 
# Configure logging
logging.basicConfig(level=logging.DEBUG)
 
async def main():
    # Load environment variables from .env file
    load_dotenv()
   
    # Set the plugins directory path
    plugins_directory = "sample_temps"
   
    # Create a new kernel instance
    kernel = sk.Kernel()
   
    # Define the service ID for the planner
    service_id = "planner"
   
    # Initialize AzureChatCompletion service
    chat_service = AzureChatCompletion(
                service_id=service_id,
                api_version="2023-05-15",  # Use the appropriate version
                deployment_name="sematic-kernel-test",
                api_key="ce2cdbcb860b41ed95723ae3aba17bc7",
                endpoint="https://cog-qpy3j27u2zsoo.openai.azure.com/"
            )
    
    # chat_service = AzureChatCompletion(
    #     service_id=service_id,
    #     env_file_path=".env" 
    # )
   
    # Add the chat service to the kernel
    kernel.add_service(chat_service)
   
    # Add custom API plugin to the kernel
    kernel.add_plugin(plugin=ApiPlugin(), plugin_name="api")
   
    # Add the KYBData plugin from the specified directory
    kernel.add_plugin(parent_directory=plugins_directory, plugin_name="KYBData")
   
# Run the main function asynchronously
if __name__ == "__main__":
    asyncio.run(main())