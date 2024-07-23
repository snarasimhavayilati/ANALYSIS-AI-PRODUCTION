import requests
import base64
import sys
import json
 
if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated
 
from semantic_kernel.functions.kernel_function_decorator import kernel_function
 
class ApiPlugin:
    """
    Description: ApiPlugin provides a set of functions to interact with a REST API.

    Usage:
        kernel.add_plugin(ApiPlugin(), plugin_name="api")
 
    Examples:
        {{api.FetchBusinessData}} => Fetch business data from the Middesk API.
    """
 
    @kernel_function(
        description="Fetch business data from Middesk API.",
        name="FetchBusinessData",
    )
    async def fetch_business_data(self) -> Annotated[dict, "The response from the Middesk API as a JSON object"]:
        url = "https://api-sandbox.middesk.com/v1/businesses"
        api_key = 'mk_test_c0b0a311eefade6cfa3f2997:'
        encoded_key = base64.b64encode(api_key.encode()).decode()
        headers = {
            "Accept": "application/json",
            "Authorization": f"Basic {encoded_key}"
        }
        data={
            "name": "Middesk Inc",
            "tin[tin]": "123410000",
            "website[url]": "https://www.middesk.com",
            "addresses[0][address_line1]": "577 Howard St",
            "addresses[0][address_line2]": "Suite 400",
            "addresses[0][city]": "San Francisco",
            "addresses[0][state]": "CA",
            "addresses[0][postal_code]": "94105"
        }
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        print("Executed Fetch Business Data")
        #response_string = json.dumps(response.text)
        return response.text
   
   
    @kernel_function(
        description="Investigate KYB data using provided schema and prompt.",
        name="InvestigateKYBData",
    )
    async def investigate_kyb_data(
        self,
        input_data: Annotated[dict, "The KYB data."],
        kernel: Annotated[object, "Kernel object for invoking the language model"],
    ) -> Annotated[str, "The compliance report generated from the KYB data"]:
        prompt = f"""
        This is Data: {input_data}.
        Act as a compliance officer checking Know Your Business (KYB).
        List any issues you see in the following categories: 'addresses', 'watchlist', 'bankruptcies', 'certifications', 'documents', 'liens', 'names', 'litigations', 'actions', 'policy_results'.
        If any category contains null, none, or an empty array, it indicates no problem.
        However, if there are any values in these categories, find the issues as you would when checking KYB and present them in paragraph or line format, not in JSON.
        """
        report = await kernel.invoke_function("chat_service.generate_text", {"input": prompt})
        print("Executed KYB Data")
        return report