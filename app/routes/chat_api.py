import asyncio
import json
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Google specific imports
from google.genai import types
from google import genai

# Local module imports
from models import OpenAIRequest
from auth import get_api_key
from message_processing import (
    create_gemini_prompt,
    create_encrypted_gemini_prompt,
    create_encrypted_full_gemini_prompt,
    ENCRYPTION_INSTRUCTIONS,
)
from api_helpers import (
    create_generation_config, # Corrected import name
    create_openai_error_response,
    execute_gemini_call,
)
from openai_handler import OpenAIDirectHandler
from project_id_discovery import discover_project_id

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    try:
        credential_manager_instance = fastapi_request.app.state.credential_manager
        OPENAI_DIRECT_SUFFIX = "-openai"
        OPENAI_SEARCH_SUFFIX = "-openaisearch"
        EXPERIMENTAL_MARKER = "-exp-"
        PAY_PREFIX = "[PAY]"
        EXPRESS_PREFIX = "[EXPRESS] " # Note the space for easier stripping
        
        # Model validation based on a predefined list has been removed as per user request.
        # The application will now attempt to use any provided model string.
        # We still need to fetch vertex_express_model_ids for the Express Mode logic.
        # vertex_express_model_ids = await get_vertex_express_models() # We'll use the prefix now

        # Updated logic for is_openai_direct_model
        is_openai_direct_model = False
        is_openai_search_model = False
        if request.model.endswith(OPENAI_DIRECT_SUFFIX) or request.model.endswith(OPENAI_SEARCH_SUFFIX):
            is_openai_search_model = request.model.endswith(OPENAI_SEARCH_SUFFIX)
            suffix_to_remove = OPENAI_SEARCH_SUFFIX if is_openai_search_model else OPENAI_DIRECT_SUFFIX
            temp_name_for_marker_check = request.model[:-len(suffix_to_remove)]
            # An OpenAI model can be prefixed with PAY, EXPRESS, or contain EXP
            if temp_name_for_marker_check.startswith(PAY_PREFIX) or \
               temp_name_for_marker_check.startswith(EXPRESS_PREFIX) or \
               EXPERIMENTAL_MARKER in temp_name_for_marker_check:
                is_openai_direct_model = True
        is_auto_model = request.model.endswith("-auto")
        is_grounded_search = request.model.endswith("-search")
        is_encrypted_model = request.model.endswith("-encrypt")
        is_encrypted_full_model = request.model.endswith("-encrypt-full")
        is_nothinking_model = request.model.endswith("-nothinking")
        is_max_thinking_model = request.model.endswith("-max")
        base_model_name = request.model # Start with the full model name

        # Determine base_model_name by stripping known prefixes and suffixes
        # Order of stripping: Prefixes first, then suffixes.
        
        is_express_model_request = False
        if base_model_name.startswith(EXPRESS_PREFIX):
            is_express_model_request = True
            base_model_name = base_model_name[len(EXPRESS_PREFIX):]

        if base_model_name.startswith(PAY_PREFIX):
            base_model_name = base_model_name[len(PAY_PREFIX):]

        # Suffix stripping (applied to the name after prefix removal)
        # This order matters if a model could have multiple (e.g. -encrypt-auto, though not currently a pattern)
        if is_openai_direct_model: # This check is based on request.model, so it's fine here
            # If it was an OpenAI direct model, its base name is request.model minus suffix.
            # We need to ensure PAY_PREFIX or EXPRESS_PREFIX are also stripped if they were part of the original.
            suffix_to_remove = OPENAI_SEARCH_SUFFIX if is_openai_search_model else OPENAI_DIRECT_SUFFIX
            temp_base_for_openai = request.model[:-len(suffix_to_remove)]
            if temp_base_for_openai.startswith(EXPRESS_PREFIX):
                temp_base_for_openai = temp_base_for_openai[len(EXPRESS_PREFIX):]
            if temp_base_for_openai.startswith(PAY_PREFIX):
                temp_base_for_openai = temp_base_for_openai[len(PAY_PREFIX):]
            base_model_name = temp_base_for_openai # Assign the fully stripped name
        elif is_auto_model: base_model_name = base_model_name[:-len("-auto")]
        elif is_grounded_search: base_model_name = base_model_name[:-len("-search")]
        elif is_encrypted_full_model: base_model_name = base_model_name[:-len("-encrypt-full")] # Must be before -encrypt
        elif is_encrypted_model: base_model_name = base_model_name[:-len("-encrypt")]
        elif is_nothinking_model: base_model_name = base_model_name[:-len("-nothinking")]
        elif is_max_thinking_model: base_model_name = base_model_name[:-len("-max")]
        
        # # Specific model variant checks (if any remain exclusive and not covered dynamically)
        # if is_nothinking_model and not (base_model_name.startswith("gemini-2.5-flash") or base_model_name == "gemini-2.5-pro-preview-06-05"):
        #     return JSONResponse(status_code=400, content=create_openai_error_response(400, f"Model '{request.model}' (-nothinking) is only supported for models starting with 'gemini-2.5-flash' or 'gemini-2.5-pro-preview-06-05'.", "invalid_request_error"))
        # if is_max_thinking_model and not (base_model_name.startswith("gemini-2.5-flash") or base_model_name == "gemini-2.5-pro-preview-06-05"):
        #     return JSONResponse(status_code=400, content=create_openai_error_response(400, f"Model '{request.model}' (-max) is only supported for models starting with 'gemini-2.5-flash' or 'gemini-2.5-pro-preview-06-05'.", "invalid_request_error"))

        # This will now be a dictionary
        gen_config_dict = create_generation_config(request)

        if "gemini-2.5-flash" in base_model_name or "gemini-2.5-pro" in base_model_name:
            if "thinking_config" not in gen_config_dict:
                gen_config_dict["thinking_config"] = {}
            gen_config_dict["thinking_config"]["include_thoughts"] = True

        if "gemini-2.5-flash-lite" in base_model_name:
            gen_config_dict["thinking_config"]["include_thoughts"] = False

        # Create a factory function for Gemini clients that rotates keys/credentials on each call
        # This enables race mode to use different keys for each concurrent request
        express_key_manager_instance = fastapi_request.app.state.express_key_manager
        
        async def gemini_client_factory():
            """Factory function that creates a new Gemini client with rotated key/credential."""
            if is_express_model_request:
                key_tuple = express_key_manager_instance.get_express_api_key()  # Rotates keys
                if not key_tuple:
                    raise Exception("Express API key not available for Gemini client.")
                
                original_idx, key_val = key_tuple
                
                # Check if model contains "gemini-2.5-pro" or "gemini-2.5-flash" for direct URL approach
                if "gemini-2.5-pro" in base_model_name or "gemini-2.5-flash" in base_model_name:
                    project_id = await discover_project_id(key_val)
                    base_url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global"
                    client = genai.Client(
                        vertexai=True,
                        api_key=key_val,
                        http_options=types.HttpOptions(base_url=base_url)
                    )
                    client._api_client._http_options.api_version = None
                    print(f"INFO: [ClientFactory] Created Gemini Express client for project {project_id} (key index: {original_idx})")
                    return client
                else:
                    client = genai.Client(vertexai=True, api_key=key_val)
                    print(f"INFO: [ClientFactory] Created Gemini Express client (key index: {original_idx})")
                    return client
            else:
                # SA credential path
                rotated_credentials, rotated_project_id = credential_manager_instance.get_credentials()  # Rotates credentials
                if not rotated_credentials or not rotated_project_id:
                    raise Exception("SA credentials not available for Gemini client.")
                
                client = genai.Client(vertexai=True, credentials=rotated_credentials, project=rotated_project_id, location="global")
                print(f"INFO: [ClientFactory] Created Gemini SA client for project: {rotated_project_id}")
                return client

        # Validate that we have credentials/keys available before proceeding
        if not is_openai_direct_model:
            if is_express_model_request:
                if express_key_manager_instance.get_total_keys() == 0:
                    error_msg = f"Model '{request.model}' is an Express model and requires an Express API key, but none are configured."
                    print(f"ERROR: {error_msg}")
                    return JSONResponse(status_code=401, content=create_openai_error_response(401, error_msg, "authentication_error"))
                print(f"INFO: Gemini Express Mode enabled for model: {request.model} (base: {base_model_name})")
            else:
                # Verify SA credentials exist
                test_creds, test_proj = credential_manager_instance.get_credentials()
                if not test_creds or not test_proj:
                    error_msg = f"Model '{request.model}' requires SA credentials for Gemini, but none are available or loaded."
                    print(f"ERROR: {error_msg}")
                    return JSONResponse(status_code=401, content=create_openai_error_response(401, error_msg, "authentication_error"))
                print(f"INFO: Gemini SA Mode enabled for model: {request.model}")

        if is_openai_direct_model:
            # Use the new OpenAI handler
            if is_express_model_request:
                openai_handler = OpenAIDirectHandler(express_key_manager=express_key_manager_instance)
                return await openai_handler.process_request(request, base_model_name, is_express=True, is_openai_search=is_openai_search_model)
            else:
                openai_handler = OpenAIDirectHandler(credential_manager=credential_manager_instance)
                return await openai_handler.process_request(request, base_model_name, is_openai_search=is_openai_search_model)
        elif is_auto_model:
            print(f"Processing auto model: {request.model}")
            attempts = [
                {"name": "base", "model": base_model_name, "prompt_func": create_gemini_prompt, "config_modifier": lambda c: c},
                {"name": "encrypt", "model": base_model_name, "prompt_func": create_encrypted_gemini_prompt, "config_modifier": lambda c: {**c, "system_instruction": ENCRYPTION_INSTRUCTIONS}},
                {"name": "old_format", "model": base_model_name, "prompt_func": create_encrypted_full_gemini_prompt, "config_modifier": lambda c: c}
            ]
            last_err = None
            for attempt in attempts:
                print(f"Auto-mode attempting: '{attempt['name']}' for model {attempt['model']}")
                # Apply modifier to the dictionary. Ensure modifier returns a dict.
                current_gen_config_dict = attempt["config_modifier"](gen_config_dict.copy())
                try:
                    # Pass is_auto_attempt=True for auto-mode calls
                    # For auto mode, create one client and use it (not a factory)
                    auto_client = await gemini_client_factory()
                    result = await execute_gemini_call(auto_client, attempt["model"], attempt["prompt_func"], current_gen_config_dict, request, is_auto_attempt=True)
                    return result
                except Exception as e_auto:
                    last_err = e_auto
                    print(f"Auto-attempt '{attempt['name']}' for model {attempt['model']} failed: {e_auto}")
                    await asyncio.sleep(1)
            
            print(f"All auto attempts failed. Last error: {last_err}")
            err_msg = f"All auto-mode attempts failed for model {request.model}. Last error: {str(last_err)}"
            if not request.stream and last_err:
                 return JSONResponse(status_code=500, content=create_openai_error_response(500, err_msg, "server_error"))
            elif request.stream:
                # This is the final error handling for auto-mode if all attempts fail AND it was a streaming request
                async def final_auto_error_stream():
                    err_content = create_openai_error_response(500, err_msg, "server_error")
                    json_payload_final_auto_error = json.dumps(err_content)
                    # Log the final error being sent to client after all auto-retries failed
                    print(f"DEBUG: Auto-mode all attempts failed. Yielding final error JSON: {json_payload_final_auto_error}")
                    yield f"data: {json_payload_final_auto_error}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(final_auto_error_stream(), media_type="text/event-stream")
            return JSONResponse(status_code=500, content=create_openai_error_response(500, "All auto-mode attempts failed without specific error.", "server_error"))

        else: # Not an auto model
            current_prompt_func = create_gemini_prompt
            # Determine the actual model string to call the API with (e.g., "gemini-1.5-pro-search")

            if is_grounded_search:
                search_tool = types.Tool(google_search=types.GoogleSearch())
                # Add or update the 'tools' key in the gen_config_dict
                if "tools" in gen_config_dict and isinstance(gen_config_dict["tools"], list):
                    gen_config_dict["tools"].append(search_tool)
                else:
                    gen_config_dict["tools"] = [search_tool]
            
            # For encrypted models, system instructions are handled by the prompt_func
            elif is_encrypted_model:
                current_prompt_func = create_encrypted_gemini_prompt
            elif is_encrypted_full_model:
                current_prompt_func = create_encrypted_full_gemini_prompt
            
            # For -nothinking or -max, the thinking_config is already set in create_generation_config
            # or can be adjusted here if needed, but it's part of the dictionary.
            # Example: if is_nothinking_model: gen_config_dict["thinking_config"] = {"thinking_budget": 0}
            # This is already handled by create_generation_config based on current logic.
            # If specific overrides are needed here, they would modify gen_config_dict.
            if is_nothinking_model or is_max_thinking_model:
                if is_nothinking_model:
                    budget = 128 if "gemini-2.5-pro" in base_model_name else 0
                else:  # is_max_thinking_model
                    budget = 32768 if "gemini-2.5-pro" in base_model_name else 24576

                # Ensure thinking_config is a dictionary before updating
                if not isinstance(gen_config_dict.get("thinking_config"), dict):
                    gen_config_dict["thinking_config"] = {}
                gen_config_dict["thinking_config"]["thinking_budget"] = budget
                if "gemini-2.5-flash-lite" in base_model_name and is_max_thinking_model:
                    gen_config_dict["thinking_config"]["include_thoughts"] = True
                if budget == 0:
                    gen_config_dict["thinking_config"]["include_thoughts"] = False

            return await execute_gemini_call(gemini_client_factory, base_model_name, current_prompt_func, gen_config_dict, request)

    except Exception as e:
        error_msg = f"Unexpected error in chat_completions endpoint: {str(e)}"
        print(error_msg)
        return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))
