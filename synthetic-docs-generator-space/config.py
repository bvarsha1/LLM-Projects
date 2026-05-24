import os
from dotenv import load_dotenv
import litellm

load_dotenv()
os.environ["LITELLM_LOOKUP_CACHE"] = "False"

def azure_completion_client(is_planner: bool = False, messages: list = None, response_format = None):
    """
    Unified client wrapper that dynamically reads your active 
    Deployment, Model, and API Version parameters directly from .env
    """
    if messages is None:
        messages = []

    # Read endpoint and authorization keys from the environment
    api_base = os.getenv("AZURE_AI_ENDPOINT") or os.getenv("AZURE_API_BASE")
    api_key = os.getenv("AZURE_AI_API_KEY") or os.getenv("AZURE_API_KEY")
    api_version = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")

    # Select deployment string based on orchestration pipeline role
    if is_planner:
        deployment_name = os.getenv("AZURE_PLANNER_DEPLOYMENT", "gpt-4.1-mini")
    else:
        deployment_name = os.getenv("AZURE_WRITER_DEPLOYMENT", "gpt-4.1-nano")

    # LiteLLM routing rule: should use 'azure/<deployment_name>' format for Azure instances
    target_model_route = f"azure/{deployment_name}"

    # Fire execution call out directly to the Azure AI Foundry resource
    response = litellm.completion(
        model=target_model_route,
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        api_version=api_version,
        response_format=response_format
    )
    
    return response