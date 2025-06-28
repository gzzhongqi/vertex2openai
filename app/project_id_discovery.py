import aiohttp
import json
import re
from typing import Dict, Optional
import config

# Global cache for project IDs: {api_key: project_id}
PROJECT_ID_CACHE: Dict[str, str] = {}


def _get_proxy_url() -> Optional[str]:
    """Get proxy URL from config."""
    return config.PROXY_URL


async def discover_project_id(api_key: str) -> str:
    """
    Discover project ID by triggering an intentional error with a non-existent model.
    The project ID is extracted from the error message and cached for future use.
    
    Args:
        api_key: The Vertex AI Express API key
        
    Returns:
        The discovered project ID
        
    Raises:
        Exception: If project ID discovery fails
    """
    # Check cache first
    if api_key in PROJECT_ID_CACHE:
        print(f"INFO: Using cached project ID: {PROJECT_ID_CACHE[api_key]}")
        return PROJECT_ID_CACHE[api_key]
    
    # Use a non-existent model to trigger error
    error_url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.7-pro-preview-05-06:streamGenerateContent?key={api_key}"
    
    # Create minimal request payload
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "test"}]}]
    }
    
    proxy = _get_proxy_url()
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(error_url, json=payload, proxy=proxy, ssl=getattr(config, "SSL_CERT_FILE", None)) as response:
                response_text = await response.text()
                
                try:
                    # Try to parse as JSON first
                    error_data = json.loads(response_text)
                    
                    # Handle array response format
                    if isinstance(error_data, list) and len(error_data) > 0:
                        error_data = error_data[0]
                    
                    if "error" in error_data:
                        error_message = error_data["error"].get("message", "")
                        # Extract project ID from error message
                        # Pattern: "projects/39982734461/locations/..."
                        match = re.search(r'projects/(\d+)/locations/', error_message)
                        if match:
                            project_id = match.group(1)
                            PROJECT_ID_CACHE[api_key] = project_id
                            print(f"INFO: Discovered project ID: {project_id}")
                            return project_id
                except json.JSONDecodeError:
                    # If not JSON, try to find project ID in raw text
                    match = re.search(r'projects/(\d+)/locations/', response_text)
                    if match:
                        project_id = match.group(1)
                        PROJECT_ID_CACHE[api_key] = project_id
                        print(f"INFO: Discovered project ID from raw response: {project_id}")
                        return project_id
                
                raise Exception(f"Failed to discover project ID. Status: {response.status}, Response: {response_text[:500]}")
                
        except Exception as e:
            print(f"ERROR: Failed to discover project ID: {e}")
            raise