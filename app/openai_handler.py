"""
OpenAI handler module for creating clients and processing OpenAI Direct mode responses.
This module encapsulates all OpenAI-specific logic that was previously in chat_api.py.
"""
import json
import time
import httpx
from typing import Dict, Any, AsyncGenerator

from fastapi.responses import JSONResponse, StreamingResponse
import openai

from models import OpenAIRequest
from config import VERTEX_REASONING_TAG
import config as app_config
from api_helpers import (
    create_openai_error_response,
    openai_fake_stream_generator,
    StreamingReasoningProcessor
)
from message_processing import extract_reasoning_by_tags
from credentials_manager import _refresh_auth
from project_id_discovery import discover_project_id


# Wrapper classes to mimic OpenAI SDK responses for direct httpx calls
class FakeChatCompletionChunk:
    """A fake ChatCompletionChunk to wrap the dictionary from a direct API stream."""
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def model_dump(self, exclude_unset=True, exclude_none=True) -> Dict[str, Any]:
        return self._data

class FakeChatCompletion:
    """A fake ChatCompletion to wrap the dictionary from a direct non-streaming API call."""
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def model_dump(self, exclude_unset=True, exclude_none=True) -> Dict[str, Any]:
        return self._data

class ExpressClientWrapper:
    """
    A wrapper that mimics the openai.AsyncOpenAI client interface but uses direct
    httpx calls for Vertex AI Express Mode. This allows it to be used with the
    existing response handling logic.
    """
    def __init__(self, project_id: str, api_key: str, location: str = "global"):
        self.project_id = project_id
        self.api_key = api_key
        self.location = location
        self.base_url = f"https://aiplatform.googleapis.com/v1beta1/projects/{self.project_id}/locations/{self.location}/endpoints/openapi"
        
        # The 'chat.completions' structure mimics the real OpenAI client
        self.chat = self
        self.completions = self

    async def _stream_generator(self, response: httpx.Response) -> AsyncGenerator[FakeChatCompletionChunk, None]:
        """Processes the SSE stream from httpx and yields fake chunk objects."""
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                json_str = line[len("data: "):].strip()
                if json_str == "[DONE]":
                    break
                try:
                    data = json.loads(json_str)
                    yield FakeChatCompletionChunk(data)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from stream line: {json_str}")
                    continue

    async def _streaming_create(self, **kwargs) -> AsyncGenerator[FakeChatCompletionChunk, None]:
        """Handles the creation of a streaming request using httpx."""
        endpoint = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        
        payload = kwargs.copy()
        if 'extra_body' in payload:
            payload.update(payload.pop('extra_body'))

        proxies = None
        if app_config.PROXY_URL:
            if app_config.PROXY_URL.startswith("socks"):
                proxies = {"all://": app_config.PROXY_URL}
            else:
                proxies = {"https://": app_config.PROXY_URL}

        client_args = {'timeout': 300}
        if proxies:
            client_args['proxies'] = proxies
        if app_config.SSL_CERT_FILE:
            client_args['verify'] = app_config.SSL_CERT_FILE
        async with httpx.AsyncClient(**client_args) as client:
            async with client.stream("POST", endpoint, headers=headers, params=params, json=payload, timeout=None) as response:
                response.raise_for_status()
                async for chunk in self._stream_generator(response):
                    yield chunk

    async def create(self, **kwargs) -> Any:
        """
        Mimics the 'create' method of the OpenAI client.
        It builds and sends a direct HTTP request using httpx, delegating
        to the appropriate streaming or non-streaming handler.
        """
        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            return self._streaming_create(**kwargs)
        
        # Non-streaming logic
        endpoint = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        
        payload = kwargs.copy()
        if 'extra_body' in payload:
            payload.update(payload.pop('extra_body'))

        proxies = None
        if app_config.PROXY_URL:
            if app_config.PROXY_URL.startswith("socks"):
                proxies = {"all://": app_config.PROXY_URL}
            else:
                proxies = {"https://": app_config.PROXY_URL}

        client_args = {'timeout': 300}
        if proxies:
            client_args['proxies'] = proxies
        if app_config.SSL_CERT_FILE:
            client_args['verify'] = app_config.SSL_CERT_FILE
        async with httpx.AsyncClient(**client_args) as client:
            response = await client.post(endpoint, headers=headers, params=params, json=payload, timeout=None)
            response.raise_for_status()
            return FakeChatCompletion(response.json())


class OpenAIDirectHandler:
    """Handles OpenAI Direct mode operations including client creation and response processing."""
    
    def __init__(self, credential_manager=None, express_key_manager=None):
        self.credential_manager = credential_manager
        self.express_key_manager = express_key_manager
        safety_threshold = "BLOCK_NONE" if app_config.SAFETY_SCORE else "OFF"
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": safety_threshold},
            {"category": 'HARM_CATEGORY_CIVIC_INTEGRITY', "threshold": safety_threshold}
        ]

    def create_openai_client(self, project_id: str, gcp_token: str, location: str = "global") -> openai.AsyncOpenAI:
        """Create an OpenAI client configured for Vertex AI endpoint."""
        endpoint_url = (
            f"https://aiplatform.googleapis.com/v1beta1/"
            f"projects/{project_id}/locations/{location}/endpoints/openapi"
        )
        
        proxies = None
        if app_config.PROXY_URL:
            if app_config.PROXY_URL.startswith("socks"):
                proxies = {"all://": app_config.PROXY_URL}
            else:
                proxies = {"https://": app_config.PROXY_URL}

        client_args = {}
        if proxies:
            client_args['proxies'] = proxies
        if app_config.SSL_CERT_FILE:
            client_args['verify'] = app_config.SSL_CERT_FILE
        
        http_client = httpx.AsyncClient(**client_args) if client_args else None
        return openai.AsyncOpenAI(
            base_url=endpoint_url,
            api_key=gcp_token,  # OAuth token
            http_client=http_client,
        )
    
    def prepare_openai_params(self, request: OpenAIRequest, model_id: str, is_openai_search: bool = False) -> Dict[str, Any]:
        """
        Prepare parameters for OpenAI API call by converting the request to a dictionary,
        and then overriding the model. This is more robust than manually picking parameters.
        """
        # Convert the request to a dict, excluding unset values. `None` values inside
        # nested models (like messages) are preserved.
        params = request.model_dump(exclude_unset=True)
        
        # Update model and filter out top-level None values.
        params['model'] = model_id
        
        if is_openai_search:
            params['web_search_options'] = {}
            
        openai_params = {k: v for k, v in params.items() if v is not None}
        if "reasoning_effort" in openai_params and openai_params["reasoning_effort"] not in ["low", "medium", "high"]:
            del openai_params["reasoning_effort"]
        return openai_params
    
    
    def prepare_extra_body(self) -> Dict[str, Any]:
        """Prepare extra body parameters for OpenAI API call."""
        return {
            "extra_body": {
                'google': {
                    'safety_settings': self.safety_settings,
                    'thought_tag_marker': VERTEX_REASONING_TAG,
                    "thinking_config": {
                        "include_thoughts": True
                    }
                }
            }
        }
    
    async def handle_streaming_response(
        self,
        openai_client: Any, # Can be openai.AsyncOpenAI or our wrapper
        openai_params: Dict[str, Any],
        openai_extra_body: Dict[str, Any],
        request: OpenAIRequest
    ) -> StreamingResponse:
        """Handle streaming responses for OpenAI Direct mode."""
        if app_config.FAKE_STREAMING_ENABLED:
            print(f"INFO: OpenAI Fake Streaming (SSE Simulation) ENABLED for model '{request.model}'.")
            return StreamingResponse(
                openai_fake_stream_generator(
                    openai_client=openai_client,
                    openai_params=openai_params,
                    openai_extra_body=openai_extra_body,
                    request_obj=request,
                    is_auto_attempt=False
                ),
                media_type="text/event-stream"
            )
        else:
            print(f"INFO: OpenAI True Streaming ENABLED for model '{request.model}'.")
            return StreamingResponse(
                self._true_stream_generator(openai_client, openai_params, openai_extra_body, request),
                media_type="text/event-stream"
            )
    
    async def _true_stream_generator(
        self,
        openai_client: Any, # Can be openai.AsyncOpenAI or our wrapper
        openai_params: Dict[str, Any],
        openai_extra_body: Dict[str, Any],
        request: OpenAIRequest
    ) -> AsyncGenerator[str, None]:
        """Generate true streaming response."""
        try:
            # Ensure stream=True is explicitly passed for real streaming
            openai_params_for_stream = {**openai_params, "stream": True}
            stream_response = await openai_client.chat.completions.create(
                **openai_params_for_stream,
                extra_body=openai_extra_body
            )
            
            # Create processor for tag-based extraction across chunks
            reasoning_processor = StreamingReasoningProcessor(VERTEX_REASONING_TAG)
            chunk_count = 0
            has_sent_content = False
            
            async for chunk in stream_response:
                chunk_count += 1
                try:
                    chunk_as_dict = chunk.model_dump(exclude_unset=True, exclude_none=True)
                    
                    choices = chunk_as_dict.get('choices')
                    if choices and isinstance(choices, list) and len(choices) > 0:
                        delta = choices[0].get('delta')
                        if delta and isinstance(delta, dict):
                            # Always remove extra_content if present
                            
                            if 'extra_content' in delta:
                                del delta['extra_content']
                            
                            content = delta.get('content', '')
                            if content:
                                # Use the processor to extract reasoning
                                processed_content, current_reasoning = reasoning_processor.process_chunk(content)
                                
                                # Send chunks for both reasoning and content as they arrive
                                original_choice = chunk_as_dict['choices'][0]
                                original_finish_reason = original_choice.get('finish_reason')
                                original_usage = original_choice.get('usage')

                                if current_reasoning:
                                    reasoning_delta = {'reasoning_content': current_reasoning}
                                    reasoning_payload = {
                                        "id": chunk_as_dict["id"], "object": chunk_as_dict["object"],
                                        "created": chunk_as_dict["created"], "model": chunk_as_dict["model"],
                                        "choices": [{"index": 0, "delta": reasoning_delta, "finish_reason": None}]
                                    }
                                    yield f"data: {json.dumps(reasoning_payload)}\n\n"
                                
                                if processed_content:
                                    content_delta = {'content': processed_content}
                                    finish_reason_for_this_content_delta = None
                                    usage_for_this_content_delta = None

                                    if original_finish_reason and not reasoning_processor.inside_tag:
                                        finish_reason_for_this_content_delta = original_finish_reason
                                        if original_usage:
                                            usage_for_this_content_delta = original_usage
                                    
                                    content_payload = {
                                        "id": chunk_as_dict["id"], "object": chunk_as_dict["object"],
                                        "created": chunk_as_dict["created"], "model": chunk_as_dict["model"],
                                        "choices": [{"index": 0, "delta": content_delta, "finish_reason": finish_reason_for_this_content_delta}]
                                    }
                                    if usage_for_this_content_delta:
                                        content_payload['choices'][0]['usage'] = usage_for_this_content_delta
                                    
                                    yield f"data: {json.dumps(content_payload)}\n\n"
                                    has_sent_content = True
                                
                            elif original_choice.get('finish_reason'): # Check original_choice for finish_reason
                                yield f"data: {json.dumps(chunk_as_dict)}\n\n"
                            elif not content and not original_choice.get('finish_reason') :
                                yield f"data: {json.dumps(chunk_as_dict)}\n\n"
                    else:
                        # Yield chunks without choices too (they might contain metadata)
                        yield f"data: {json.dumps(chunk_as_dict)}\n\n"

                except Exception as chunk_error:
                    error_msg = f"Error processing OpenAI chunk for {request.model}: {str(chunk_error)}"
                    print(f"ERROR: {error_msg}")
                    if len(error_msg) > 1024:
                        error_msg = error_msg[:1024] + "..."
                    error_response = create_openai_error_response(500, error_msg, "server_error")
                    yield f"data: {json.dumps(error_response)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            
            # Debug logging for buffer state and chunk count
            # print(f"DEBUG: Stream ended after {chunk_count} chunks. Buffer state - tag_buffer: '{reasoning_processor.tag_buffer}', "
            #       f"inside_tag: {reasoning_processor.inside_tag}, "
            #       f"reasoning_buffer: '{reasoning_processor.reasoning_buffer[:50]}...' if reasoning_processor.reasoning_buffer else ''")
            # Flush any remaining buffered content
            remaining_content, remaining_reasoning = reasoning_processor.flush_remaining()
            
            # Send any remaining reasoning first
            if remaining_reasoning:
                reasoning_flush_payload = {
                    "id": f"chatcmpl-flush-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"reasoning_content": remaining_reasoning}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(reasoning_flush_payload)}\n\n"
            
            # Send any remaining content
            if remaining_content:
                content_flush_payload = {
                    "id": f"chatcmpl-flush-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": remaining_content}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(content_flush_payload)}\n\n"
                has_sent_content = True
            
            # Always send a finish reason chunk
            finish_payload = {
                "id": f"chatcmpl-final-{int(time.time())}", # Kilo Code: Changed ID for clarity
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(finish_payload)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as stream_error:
            error_msg = str(stream_error)
            if len(error_msg) > 1024:
                error_msg = error_msg[:1024] + "..."
            error_msg_full = f"Error during OpenAI streaming for {request.model}: {error_msg}"
            print(f"ERROR: {error_msg_full}")
            error_response = create_openai_error_response(500, error_msg_full, "server_error")
            yield f"data: {json.dumps(error_response)}\n\n"
            yield "data: [DONE]\n\n"
    
    async def handle_non_streaming_response(
        self,
        openai_client: Any, # Can be openai.AsyncOpenAI or our wrapper
        openai_params: Dict[str, Any],
        openai_extra_body: Dict[str, Any],
        request: OpenAIRequest
    ) -> JSONResponse:
        """Handle non-streaming responses for OpenAI Direct mode."""
        try:
            # Ensure stream=False is explicitly passed
            openai_params_non_stream = {**openai_params, "stream": False}
            response = await openai_client.chat.completions.create(
                **openai_params_non_stream,
                extra_body=openai_extra_body
            )
            response_dict = response.model_dump(exclude_unset=True, exclude_none=True)
            
            try:
                choices = response_dict.get('choices')
                if choices and isinstance(choices, list) and len(choices) > 0:
                    message_dict = choices[0].get('message')
                    if message_dict and isinstance(message_dict, dict):
                        # Always remove extra_content from the message if it exists
                        if 'extra_content' in message_dict:
                            del message_dict['extra_content']
                        
                        # Extract reasoning from content
                        full_content = message_dict.get('content')
                        actual_content = full_content if isinstance(full_content, str) else ""
                        
                        if actual_content:
                            print(f"INFO: OpenAI Direct Non-Streaming - Applying tag extraction with fixed marker: '{VERTEX_REASONING_TAG}'")
                            reasoning_text, actual_content = extract_reasoning_by_tags(actual_content, VERTEX_REASONING_TAG)
                            message_dict['content'] = actual_content
                            if reasoning_text:
                                message_dict['reasoning_content'] = reasoning_text
                                # print(f"DEBUG: Tag extraction success. Reasoning len: {len(reasoning_text)}, Content len: {len(actual_content)}")
                            # else:
                            #     print(f"DEBUG: No content found within fixed tag '{VERTEX_REASONING_TAG}'.")
                        else:
                            print(f"WARNING: OpenAI Direct Non-Streaming - No initial content found in message.")
                            message_dict['content'] = ""
                            
            except Exception as e_reasoning:
                print(f"WARNING: Error during non-streaming reasoning processing for model {request.model}: {e_reasoning}")
            
            return JSONResponse(content=response_dict)
            
        except Exception as e:
            error_msg = f"Error calling OpenAI client for {request.model}: {str(e)}"
            print(f"ERROR: {error_msg}")
            return JSONResponse(
                status_code=500, 
                content=create_openai_error_response(500, error_msg, "server_error")
            )
    
    async def process_request(self, request: OpenAIRequest, base_model_name: str, is_express: bool = False, is_openai_search: bool = False):
        """Main entry point for processing OpenAI Direct mode requests."""
        print(f"INFO: Using OpenAI Direct Path for model: {request.model} (Express: {is_express})")
        
        client: Any = None # Can be openai.AsyncOpenAI or our wrapper

        try:
            if is_express:
                if not self.express_key_manager:
                    raise Exception("Express mode requires an ExpressKeyManager, but it was not provided.")
                
                key_tuple = self.express_key_manager.get_express_api_key()
                if not key_tuple:
                    raise Exception("OpenAI Express Mode requires an API key, but none were available.")
                
                _, express_api_key = key_tuple
                project_id = await discover_project_id(express_api_key)
                
                client = ExpressClientWrapper(project_id=project_id, api_key=express_api_key)
                print(f"INFO: [OpenAI Express Path] Using ExpressClientWrapper for project: {project_id}")

            else: # Standard SA-based OpenAI SDK Path
                if not self.credential_manager:
                    raise Exception("Standard OpenAI Direct mode requires a CredentialManager.")

                rotated_credentials, rotated_project_id = self.credential_manager.get_credentials()
                if not rotated_credentials or not rotated_project_id:
                    raise Exception("OpenAI Direct Mode requires GCP credentials, but none were available.")

                print(f"INFO: [OpenAI Direct Path] Using credentials for project: {rotated_project_id}")
                gcp_token = _refresh_auth(rotated_credentials)
                if not gcp_token:
                    raise Exception(f"Failed to obtain valid GCP token for OpenAI client (Project: {rotated_project_id}).")
                client = self.create_openai_client(rotated_project_id, gcp_token)

            model_id = f"google/{base_model_name}"
            openai_params = self.prepare_openai_params(request, model_id, is_openai_search)
            openai_extra_body = self.prepare_extra_body()
            
            if request.stream:
                return await self.handle_streaming_response(
                    client, openai_params, openai_extra_body, request
                )
            else:
                return await self.handle_non_streaming_response(
                    client, openai_params, openai_extra_body, request
                )
        except Exception as e:
            error_msg = f"Error in process_request for {request.model}: {e}"
            print(f"ERROR: {error_msg}")
            return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))