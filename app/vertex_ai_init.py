import json
from typing import Optional
from google import genai
from credentials_manager import CredentialManager, parse_multiple_json_credentials
import config as app_config
from google.genai import types
from model_loader import refresh_models_config_cache # Import new model loader function

# VERTEX_EXPRESS_MODELS list is now dynamically loaded via model_loader
# The constant VERTEX_EXPRESS_MODELS previously defined here is removed.
# Consumers should use get_vertex_express_models() from model_loader.

# Global 'client' and 'get_vertex_client()' are removed.

def _get_http_options() -> Optional[types.HttpOptions]:
    """Get http options from config."""
    if app_config.PROXY_URL and app_config.PROXY_URL.startswith("socks"):
        return types.HttpOptions(
            client_args={'proxy': app_config.PROXY_URL},
            async_client_args={'proxy': app_config.PROXY_URL},
        )
    return None


async def init_vertex_ai(credential_manager_instance: CredentialManager) -> bool: # Made async
    """
    Initializes the credential manager with credentials from GOOGLE_CREDENTIALS_JSON (if provided)
    and verifies if any credentials (environment or file-based through the manager) are available.
    The CredentialManager itself handles loading file-based credentials upon its instantiation.
    This function primarily focuses on augmenting the manager with env var credentials.

    Returns True if any credentials seem available in the manager, False otherwise.
    """
    try:
        credentials_json_str = app_config.GOOGLE_CREDENTIALS_JSON_STR
        env_creds_loaded_into_manager = False

        if credentials_json_str:
            print("INFO: Found GOOGLE_CREDENTIALS_JSON environment variable. Attempting to load into CredentialManager.")
            try:
                # Attempt 1: Parse as multiple JSON objects
                json_objects = parse_multiple_json_credentials(credentials_json_str)
                if json_objects:
                    print(f"DEBUG: Parsed {len(json_objects)} potential credential objects from GOOGLE_CREDENTIALS_JSON.")
                    success_count = credential_manager_instance.load_credentials_from_json_list(json_objects)
                    if success_count > 0:
                        print(f"INFO: Successfully loaded {success_count} credentials from GOOGLE_CREDENTIALS_JSON into manager.")
                        env_creds_loaded_into_manager = True
                
                # Attempt 2: If multiple parsing/loading didn't add any, try parsing/loading as a single JSON object
                if not env_creds_loaded_into_manager:
                    print("DEBUG: Multi-JSON loading from GOOGLE_CREDENTIALS_JSON did not add to manager or was empty. Attempting single JSON load.")
                    try:
                        credentials_info = json.loads(credentials_json_str)
                        # Basic validation (CredentialManager's add_credential_from_json does more thorough validation)

                        if isinstance(credentials_info, dict) and \
                           all(field in credentials_info for field in ["type", "project_id", "private_key_id", "private_key", "client_email"]):
                            if credential_manager_instance.add_credential_from_json(credentials_info):
                                print("INFO: Successfully loaded single credential from GOOGLE_CREDENTIALS_JSON into manager.")
                                # env_creds_loaded_into_manager = True # Redundant, as this block is conditional on it being False
                            else:
                                print("WARNING: Single JSON from GOOGLE_CREDENTIALS_JSON failed to load into manager via add_credential_from_json.")
                        else:
                             print("WARNING: Single JSON from GOOGLE_CREDENTIALS_JSON is not a valid dict or missing required fields for basic check.")
                    except json.JSONDecodeError as single_json_err:
                        print(f"WARNING: GOOGLE_CREDENTIALS_JSON could not be parsed as a single JSON object: {single_json_err}.")
                    except Exception as single_load_err:
                        print(f"WARNING: Error trying to load single JSON from GOOGLE_CREDENTIALS_JSON into manager: {single_load_err}.")
            except Exception as e_json_env:
                # This catches errors from parse_multiple_json_credentials or load_credentials_from_json_list
                print(f"WARNING: Error processing GOOGLE_CREDENTIALS_JSON env var: {e_json_env}.")
        else:
            print("INFO: GOOGLE_CREDENTIALS_JSON environment variable not found.")

        # Attempt to pre-warm the model configuration cache
        print("INFO: Attempting to pre-warm model configuration cache during startup...")
        models_loaded_successfully = await refresh_models_config_cache()
        if models_loaded_successfully:
            print("INFO: Model configuration cache pre-warmed successfully.")
        else:
            print("WARNING: Failed to pre-warm model configuration cache during startup. It will be loaded lazily on first request.")
            # We don't necessarily fail the entire init_vertex_ai if model list fetching fails,
            # as credential validation might still be important, and model list can be fetched later.

        # CredentialManager's __init__ calls load_credentials_list() for files.
        # refresh_credentials_list() re-scans files and combines with in-memory (already includes env creds if loaded above).
        # The return value of refresh_credentials_list indicates if total > 0
        if credential_manager_instance.refresh_credentials_list():
            total_creds = credential_manager_instance.get_total_credentials()
            print(f"INFO: Credential Manager reports {total_creds} credential(s) available (from files and/or GOOGLE_CREDENTIALS_JSON).")
            
            # Optional: Attempt to validate one of the credentials by creating a temporary client.
            # This adds a check that at least one credential is functional.
            print("INFO: Attempting to validate a credential by creating a temporary client...")
            temp_creds_val, temp_project_id_val = credential_manager_instance.get_credentials()
            if temp_creds_val and temp_project_id_val:
                try:
                    http_options = _get_http_options()
                    if http_options:
                        _ = genai.Client(vertexai=True, credentials=temp_creds_val, project=temp_project_id_val, location="global", http_options=http_options)
                    else:
                        _ = genai.Client(vertexai=True, credentials=temp_creds_val, project=temp_project_id_val, location="global")
                    print(f"INFO: Successfully validated a credential from Credential Manager (Project: {temp_project_id_val}). Initialization check passed.")
                    return True
                except Exception as e_val:
                    print(f"WARNING: Failed to validate a random credential from manager by creating a temp client: {e_val}. App may rely on non-validated credentials.")
                    # Still return True if credentials exist, as the app might still function with other valid credentials.
                    # The per-request client creation will be the ultimate test for a specific credential.
                    return True # Credentials exist, even if one failed validation here.
            elif total_creds > 0 : # Credentials listed but get_random_credentials returned None
                 print(f"WARNING: {total_creds} credentials reported by manager, but could not retrieve one for validation. Problems might occur.")
                 return True # Still, credentials are listed.
            else: # No creds from get_random_credentials and total_creds is 0
                 print("ERROR: No credentials available after attempting to load from all sources.")
                 return False # No credentials reported by manager and get_random_credentials gave none.
        else:
            print("ERROR: Credential Manager reports no available credentials after processing all sources.")
            return False

    except Exception as e:
        print(f"CRITICAL ERROR during Vertex AI credential setup: {e}")
        return False