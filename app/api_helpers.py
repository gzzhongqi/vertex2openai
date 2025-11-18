import json
import time
import math
import asyncio
from typing import List, Dict, Any, Callable, Union, Optional, Awaitable, AsyncGenerator

from fastapi.responses import JSONResponse, StreamingResponse
from google.auth.transport.requests import Request as AuthRequest
from google.genai import types
from openai import AsyncOpenAI 


from models import OpenAIRequest, OpenAIMessage
from message_processing import (
    convert_to_openai_format,
    convert_chunk_to_openai,
    extract_reasoning_by_tags,
    _create_safety_ratings_html
)
import config as app_config
from config import VERTEX_REASONING_TAG


async def race_async_calls(call_func: Callable, num_concurrent: int = 3):
    """
    Race multiple identical async calls and return the first successful result.
    Cancel all other tasks once the first one succeeds.
    
    Args:
        call_func: An async callable that returns a result
        num_concurrent: Number of concurrent calls to make
    
    Returns:
        The result from the first successful call
    
    Raises:
        Exception: If all concurrent calls fail, raises the last exception
    """
    tasks = [asyncio.create_task(call_func()) for _ in range(num_concurrent)]
    
    try:
        while tasks:
            # Wait for the first task to complete
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            for task in done:
                try:
                    result = task.result()
                    # First successful result - cancel all pending tasks
                    print(f"INFO: Race mode - Got successful result, cancelling {len(pending)} pending tasks")
                    for pending_task in pending:
                        pending_task.cancel()
                    # Wait a bit for cancellations to process
                    if pending:
                        await asyncio.wait(pending, timeout=0.1)
                    return result
                except Exception as e:
                    # This task failed, try the next one
                    print(f"WARNING: Race mode - One task failed: {type(e).__name__}: {str(e)[:100]}")
                    tasks.remove(task)
                    continue
            
            # Update tasks list to only include pending ones
            tasks = list(pending)
        
        # All tasks failed
        raise Exception("All concurrent race attempts failed")
    
    except asyncio.CancelledError:
        # If this race itself is cancelled, cancel all tasks
        for task in tasks:
            task.cancel()
        raise

async def race_for_longest_string(call_func: Callable, num_concurrent: int = 3):
    """
    Race multiple identical async calls, wait for all to complete, 
    and return the result with the longest content string.
    
    Args:
        call_func: An async callable that returns a result object.
        num_concurrent: Number of concurrent calls to make.
    
    Returns:
        The result from the call that produced the longest content.
        
    Raises:
        Exception: If all concurrent calls fail.
    """
    tasks = [asyncio.create_task(call_func()) for _ in range(num_concurrent)]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"WARNING: Race-for-longest - Task {i} failed: {type(result).__name__}: {str(result)[:100]}")
        else:
            successful_results.append(result)
            
    if not successful_results:
        raise Exception("All concurrent race-for-longest attempts failed.")

    longest_result = None
    max_len = -1

    for result in successful_results:
        content_len = 0
        try:
            # Heuristic to find content length for both Gemini and OpenAI Direct objects
            if hasattr(result, 'candidates') and result.candidates: # Gemini response
                content = result.candidates[0].content.parts[0].text
                content_len = len(content)
            elif hasattr(result, 'choices') and result.choices: # OpenAI-like response
                content = result.choices[0].message.content
                content_len = len(content) if content else 0
        except (AttributeError, IndexError) as e:
            print(f"WARNING: Could not determine content length for a result. Error: {e}")
            content_len = 0
            
        if content_len > max_len:
            max_len = content_len
            longest_result = result
            
    print(f"INFO: Race-for-longest - Chose result with content length: {max_len}")
    return longest_result

async def race_streaming_generators(generator_factories: List[Callable[[], AsyncGenerator]]):
    """
    Races multiple async generators and yields from the one that produces the first item the fastest.
    
    Args:
        generator_factories: A list of no-argument functions that each return an async generator.
    """
    queue = asyncio.Queue()
    tasks = []
    
    async def _runner(factory: Callable[[], AsyncGenerator], task_id: int):
        """Wraps a generator to put its first item into a queue."""
        try:
            generator = factory()
            first_item = await anext(generator)
            await queue.put((task_id, first_item, generator))
        except (StopAsyncIteration, Exception) as e:
            await queue.put((task_id, e, None))

    try:
        # Start all runners
        for i, factory in enumerate(generator_factories):
            task = asyncio.create_task(_runner(factory, i))
            tasks.append(task)
            
        # Wait for the first one to put something in the queue
        winner_id, first_item, winner_generator = await queue.get()
        
        # If the first result is an exception, we need to wait for another
        while isinstance(first_item, Exception):
            print(f"WARNING: Race-streaming - Task {winner_id} failed before yielding first chunk: {first_item}")
            tasks.pop(winner_id) # Should be careful with index
            if not tasks:
                raise Exception("All streaming race attempts failed before yielding any data.") from first_item
            
            # Reset task list to avoid index issues
            active_tasks = [t for t in tasks if not t.done()]
            if not active_tasks:
                 raise Exception("All streaming race attempts failed before yielding any data.")
            tasks = active_tasks
            
            winner_id, first_item, winner_generator = await queue.get()

        print(f"INFO: Race-streaming - Task {winner_id} was the fastest to yield data. Cancelling others.")
        
        # Cancel all other tasks
        for i, task in enumerate(tasks):
            if i != winner_id:
                task.cancel()
        
        # Yield the first item from the winner
        yield first_item
        
        # Yield the rest of the items from the winning generator
        async for item in winner_generator:
            yield item
            
    finally:
        # Final cleanup of all tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        # Drain the queue to prevent hanging
        while not queue.empty():
            queue.get_nowait()

class StreamingReasoningProcessor:
    def __init__(self, tag_name: str = VERTEX_REASONING_TAG):
        self.tag_name = tag_name
        self.open_tag = f"<{tag_name}>"
        self.close_tag = f"</{tag_name}>"
        self.tag_buffer = ""
        self.inside_tag = False
        self.reasoning_buffer = ""
        self.partial_tag_buffer = "" 

    def process_chunk(self, content: str) -> tuple[str, str]:
        if self.partial_tag_buffer:
            content = self.partial_tag_buffer + content
            self.partial_tag_buffer = ""
        self.tag_buffer += content
        processed_content = ""
        current_reasoning = ""
        while self.tag_buffer:
            if not self.inside_tag:
                open_pos = self.tag_buffer.find(self.open_tag)
                if open_pos == -1:
                    partial_match = False
                    for i in range(1, min(len(self.open_tag), len(self.tag_buffer) + 1)):
                        if self.tag_buffer[-i:] == self.open_tag[:i]:
                            partial_match = True
                            if len(self.tag_buffer) > i:
                                processed_content += self.tag_buffer[:-i]
                                self.partial_tag_buffer = self.tag_buffer[-i:]
                            else: self.partial_tag_buffer = self.tag_buffer
                            self.tag_buffer = ""
                            break
                    if not partial_match:
                        processed_content += self.tag_buffer
                        self.tag_buffer = ""
                    break
                else:
                    processed_content += self.tag_buffer[:open_pos]
                    self.tag_buffer = self.tag_buffer[open_pos + len(self.open_tag):]
                    self.inside_tag = True
            else: 
                close_pos = self.tag_buffer.find(self.close_tag)
                if close_pos == -1:
                    partial_match = False
                    for i in range(1, min(len(self.close_tag), len(self.tag_buffer) + 1)):
                        if self.tag_buffer[-i:] == self.close_tag[:i]:
                            partial_match = True
                            if len(self.tag_buffer) > i:
                                new_reasoning = self.tag_buffer[:-i]
                                self.reasoning_buffer += new_reasoning
                                if new_reasoning: current_reasoning = new_reasoning
                                self.partial_tag_buffer = self.tag_buffer[-i:]
                            else: self.partial_tag_buffer = self.tag_buffer
                            self.tag_buffer = ""
                            break
                    if not partial_match:
                        if self.tag_buffer:
                            self.reasoning_buffer += self.tag_buffer
                            current_reasoning = self.tag_buffer
                            self.tag_buffer = ""
                    break
                else:
                    final_reasoning_chunk = self.tag_buffer[:close_pos]
                    self.reasoning_buffer += final_reasoning_chunk
                    if final_reasoning_chunk: current_reasoning = final_reasoning_chunk
                    self.reasoning_buffer = "" 
                    self.tag_buffer = self.tag_buffer[close_pos + len(self.close_tag):]
                    self.inside_tag = False
        return processed_content, current_reasoning
    
    def flush_remaining(self) -> tuple[str, str]:
        remaining_content, remaining_reasoning = "", ""
        if self.partial_tag_buffer:
            remaining_content += self.partial_tag_buffer
            self.partial_tag_buffer = ""
        if not self.inside_tag:
            if self.tag_buffer: remaining_content += self.tag_buffer
        else:
            if self.reasoning_buffer: remaining_reasoning = self.reasoning_buffer
            if self.tag_buffer: remaining_content += self.tag_buffer
            self.inside_tag = False
        self.tag_buffer, self.reasoning_buffer = "", ""
        return remaining_content, remaining_reasoning

def create_openai_error_response(status_code: int, message: str, error_type: str) -> Dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "code": status_code, "param": None}}

def create_generation_config(request: OpenAIRequest) -> Dict[str, Any]:
    config: Dict[str, Any] = {} 
    if request.temperature is not None: config["temperature"] = request.temperature
    if request.max_tokens is not None: config["max_output_tokens"] = request.max_tokens
    if request.top_p is not None: config["top_p"] = request.top_p
    if request.top_k is not None: config["top_k"] = request.top_k
    if request.stop is not None: config["stop_sequences"] = request.stop
    if request.seed is not None: config["seed"] = request.seed
    if request.n is not None: config["candidate_count"] = request.n
    
    safety_threshold = "BLOCK_NONE"
    config["safety_settings"] = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_UNSPECIFIED", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_IMAGE_HATE", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_IMAGE_HARASSMENT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_JAILBREAK", threshold=safety_threshold)
    ]
    # config["thinking_config"] = {"include_thoughts": True}

    # 1. Add tools (function declarations)
    function_declarations = []
    if request.tools:
        for tool in request.tools:
            if tool.get("type") == "function":
                # func_def = tool.get("function")
                func_def = tool
                if func_def:
                    # Extract only the fields accepted by the Gemini API
                    declaration = {
                        "name": func_def.get("name"),
                        "description": func_def.get("description"),
                    }
                    # Get parameters and remove the $schema field if it exists
                    parameters = func_def.get("parameters")
                    if isinstance(parameters, dict) and "$schema" in parameters:
                        parameters = parameters.copy()
                        del parameters["$schema"]
                    if parameters is not None:
                        declaration["parameters"] = parameters

                    # Remove keys with None values to keep the payload clean
                    declaration = {k: v for k, v in declaration.items() if v is not None}
                    if declaration.get("name"):  # Ensure name exists
                        function_declarations.append(declaration)

    if function_declarations:
        config["tools"] = [{"function_declarations": function_declarations}]

    # 2. Add tool_config (based on tool_choice)
    tool_config = None
    if request.tool_choice:
        choice = request.tool_choice
        mode = None
        allowed_functions = None
        if isinstance(choice, str):
            if choice == "none":
                mode = "NONE"
            elif choice == "auto":
                mode = "AUTO"
        elif isinstance(choice, dict) and choice.get("type") == "function":
            func_name = choice.get("function", {}).get("name")
            if func_name:
                mode = "ANY"  # 'ANY' mode is used to force a specific function call
                allowed_functions = [func_name]
        
        # If a valid mode was parsed, build the tool_config
        if mode:
            config_dict = {"mode": mode}
            if allowed_functions:
                config_dict["allowed_function_names"] = allowed_functions
            tool_config = {"function_calling_config": config_dict}
    
    if tool_config:
        config["tool_config"] = tool_config
        
    return config


def is_gemini_response_valid(response: Any) -> bool:
    if response is None: return False
    if hasattr(response, 'text') and isinstance(response.text, str) and response.text.strip(): return True
    if hasattr(response, 'candidates') and response.candidates:
        for cand in response.candidates:
            if hasattr(cand, 'text') and isinstance(cand.text, str) and cand.text.strip(): return True
            if hasattr(cand, 'content') and hasattr(cand.content, 'parts') and cand.content.parts:
                for part in cand.content.parts:
                    if hasattr(part, 'function_call'): return True 
                    if hasattr(part, 'text') and isinstance(getattr(part, 'text', None), str) and getattr(part, 'text', '').strip(): return True
    return False

async def _chunk_openai_response_dict_for_sse(
    openai_response_dict: Dict[str, Any],
    response_id_override: Optional[str] = None, 
    model_name_override: Optional[str] = None
):
    resp_id = response_id_override or openai_response_dict.get("id", f"chatcmpl-fakestream-{int(time.time())}")
    model_name = model_name_override or openai_response_dict.get("model", "unknown")
    created_time = openai_response_dict.get("created", int(time.time()))
    
    choices = openai_response_dict.get("choices", [])
    if not choices: 
        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'error'}]})}\n\n"
        yield "data: [DONE]\n\n"
        return

    for choice_idx, choice in enumerate(choices): 
        message = choice.get("message", {})
        final_finish_reason = choice.get("finish_reason", "stop")

        if message.get("tool_calls"):
            tool_calls_list = message.get("tool_calls", [])
            for tc_item_idx, tool_call_item in enumerate(tool_calls_list):
                delta_tc_start = {
                    "tool_calls": [{
                        "index": tc_item_idx, 
                        "id": tool_call_item["id"],
                        "type": "function",
                        "function": {"name": tool_call_item["function"]["name"], "arguments": ""}
                    }]
                }
                yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': delta_tc_start, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.01) 

                delta_tc_args = {
                    "tool_calls": [{
                        "index": tc_item_idx,
                        "id": tool_call_item["id"], 
                        "function": {"arguments": tool_call_item["function"]["arguments"]}
                    }]
                }
                yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': delta_tc_args, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.01)
        
        elif message.get("content") is not None or message.get("reasoning_content") is not None : 
            reasoning_content = message.get("reasoning_content", "")
            actual_content = message.get("content") 

            if reasoning_content:
                delta_reasoning = {"reasoning_content": reasoning_content}
                yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': delta_reasoning, 'finish_reason': None}]})}\n\n"
                if actual_content is not None: await asyncio.sleep(0.05)

            content_to_chunk = actual_content if actual_content is not None else ""
            if actual_content is not None:
                chunk_size = max(1, math.ceil(len(content_to_chunk) / 10)) if content_to_chunk else 1
                if not content_to_chunk and not reasoning_content : 
                    yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': {'content': ''}, 'finish_reason': None}]})}\n\n"
                else:
                    for i in range(0, len(content_to_chunk), chunk_size):
                        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': {'content': content_to_chunk[i:i+chunk_size]}, 'finish_reason': None}]})}\n\n"
                        if len(content_to_chunk) > chunk_size: await asyncio.sleep(0.05)
        
        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': {}, 'finish_reason': final_finish_reason}]})}\n\n"

    yield "data: [DONE]\n\n"


async def gemini_fake_stream_generator( 
    client_factory: Callable[[], Awaitable[Any]], 
    model_for_api_call: str, 
    prompt_for_api_call: List[types.Content],
    gen_config_dict_for_api_call: Dict[str, Any], 
    request_obj: OpenAIRequest,
    is_auto_attempt: bool,
    race_count: int = 1
):
    print(f"FAKE STREAMING (Gemini): Prep for '{request_obj.model}' (API model string: '{model_for_api_call}', race_count: {race_count})")
    
    # Create the API call function for racing, each call gets a NEW client from the factory
    async def _make_api_call():
        gemini_client_instance = await client_factory()
        return await gemini_client_instance.aio.models.generate_content(
            model=model_for_api_call, 
            contents=prompt_for_api_call, 
            config=gen_config_dict_for_api_call
        )
    
    # Use race mode if race_count > 1
    if race_count > 1:
        api_call_task = asyncio.create_task(race_async_calls(_make_api_call, race_count))
    else:
        api_call_task = asyncio.create_task(_make_api_call())

    outer_keep_alive_interval = app_config.FAKE_STREAMING_INTERVAL_SECONDS
    if outer_keep_alive_interval > 0:
        while not api_call_task.done():
            keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": request_obj.model, "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]}
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(outer_keep_alive_interval)
    
    try:
        raw_gemini_response = await api_call_task 
        openai_response_dict = convert_to_openai_format(raw_gemini_response, request_obj.model)
        
        if hasattr(raw_gemini_response, 'prompt_feedback') and \
           hasattr(raw_gemini_response.prompt_feedback, 'block_reason') and \
           raw_gemini_response.prompt_feedback.block_reason:
            block_message = f"Response blocked by Gemini safety filter: {raw_gemini_response.prompt_feedback.block_reason}"
            if hasattr(raw_gemini_response.prompt_feedback, 'block_reason_message') and \
               raw_gemini_response.prompt_feedback.block_reason_message:
                block_message += f" (Message: {raw_gemini_response.prompt_feedback.block_reason_message})"
            raise ValueError(block_message)

        async for chunk_sse in _chunk_openai_response_dict_for_sse(
            openai_response_dict=openai_response_dict
        ):
            yield chunk_sse

    except Exception as e_outer_gemini:
        err_msg_detail = f"Error in gemini_fake_stream_generator (model: '{request_obj.model}'): {type(e_outer_gemini).__name__} - {str(e_outer_gemini)}"
        print(f"ERROR: {err_msg_detail}")
        sse_err_msg_display = str(e_outer_gemini)
        if len(sse_err_msg_display) > 512: sse_err_msg_display = sse_err_msg_display[:512] + "..."
        err_resp_sse = create_openai_error_response(500, sse_err_msg_display, "server_error")
        json_payload_error = json.dumps(err_resp_sse)
        if not is_auto_attempt:
            yield f"data: {json_payload_error}\n\n"
            yield "data: [DONE]\n\n"
        if is_auto_attempt: raise


async def openai_fake_stream_generator( 
    client_factory: Callable[[], Awaitable[Union[AsyncOpenAI, Any]]], 
    openai_params: Dict[str, Any],
    openai_extra_body: Dict[str, Any],
    request_obj: OpenAIRequest,
    is_auto_attempt: bool,
    race_count: int = 1
):
    api_model_name = openai_params.get("model", "unknown-openai-model")
    print(f"FAKE STREAMING (OpenAI Direct): Prep for '{request_obj.model}' (API model: '{api_model_name}', race_count: {race_count})")
    response_id = f"chatcmpl-openaidirectfake-{int(time.time())}"
    
    async def _openai_api_call_task():
        openai_client = await client_factory()
        params_for_call = openai_params.copy()
        params_for_call['stream'] = False 
        return await openai_client.chat.completions.create(**params_for_call, extra_body=openai_extra_body)

    # Use race mode if race_count > 1
    if race_count > 1:
        api_call_task = asyncio.create_task(race_async_calls(_openai_api_call_task, race_count))
    else:
        api_call_task = asyncio.create_task(_openai_api_call_task())
    outer_keep_alive_interval = app_config.FAKE_STREAMING_INTERVAL_SECONDS
    if outer_keep_alive_interval > 0:
        while not api_call_task.done():
            keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": request_obj.model, "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]}
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(outer_keep_alive_interval)

    try:
        raw_response_obj = await api_call_task 
        openai_response_dict = raw_response_obj.model_dump(exclude_unset=True, exclude_none=True)

        if app_config.SAFETY_SCORE and hasattr(raw_response_obj, "choices") and raw_response_obj.choices:
            for i, choice_obj in enumerate(raw_response_obj.choices):
                if hasattr(choice_obj, "safety_ratings") and choice_obj.safety_ratings:
                    safety_html = _create_safety_ratings_html(choice_obj.safety_ratings)
                    if i < len(openai_response_dict.get("choices", [])):
                        choice_dict = openai_response_dict["choices"][i]
                        message_dict = choice_dict.get("message")
                        if message_dict:
                            current_content = message_dict.get("content") or ""
                            message_dict["content"] = current_content + safety_html

        if openai_response_dict.get("choices") and \
           isinstance(openai_response_dict["choices"], list) and \
           len(openai_response_dict["choices"]) > 0:
            
            first_choice_dict_item = openai_response_dict["choices"]
            if first_choice_dict_item and isinstance(first_choice_dict_item, dict) :
                choice_message_ref = first_choice_dict_item.get("message", {})
                original_content = choice_message_ref.get("content")
                if isinstance(original_content, str):
                    reasoning_text, actual_content = extract_reasoning_by_tags(original_content, VERTEX_REASONING_TAG)
                    choice_message_ref["content"] = actual_content
                    if reasoning_text:
                        choice_message_ref["reasoning_content"] = reasoning_text
        
        async for chunk_sse in _chunk_openai_response_dict_for_sse(
            openai_response_dict=openai_response_dict,
            response_id_override=response_id, 
            model_name_override=request_obj.model
        ):
            yield chunk_sse
            
    except Exception as e_outer: 
        err_msg_detail = f"Error in openai_fake_stream_generator (model: '{request_obj.model}'): {type(e_outer).__name__} - {str(e_outer)}"
        print(f"ERROR: {err_msg_detail}")
        sse_err_msg_display = str(e_outer)
        if len(sse_err_msg_display) > 512: sse_err_msg_display = sse_err_msg_display[:512] + "..."
        err_resp_sse = create_openai_error_response(500, sse_err_msg_display, "server_error")
        json_payload_error = json.dumps(err_resp_sse)
        if not is_auto_attempt:
            yield f"data: {json_payload_error}\n\n"
            yield "data: [DONE]\n\n"
        if is_auto_attempt: raise


async def execute_gemini_call(
    client_or_factory: Union[Any, Callable[[], Awaitable[Any]]],
    model_to_call: str,  
    prompt_func: Callable[[List[OpenAIMessage]], List[types.Content]], 
    gen_config_dict: Dict[str, Any], 
    request_obj: OpenAIRequest, 
    is_auto_attempt: bool = False
):
    actual_prompt_for_call = prompt_func(request_obj.messages)

    # Normalize the client/factory input. If a client object is passed (for auto-mode),
    # wrap it in a factory-like lambda. Otherwise, use the provided factory.
    client_factory = client_or_factory if callable(client_or_factory) else (lambda: asyncio.sleep(0, result=client_or_factory))
    
    print(f"INFO: execute_gemini_call for requested API model '{model_to_call}'. Original request model: '{request_obj.model}'")
    
    # Race mode is enabled for normal calls but not for auto-mode attempts
    race_enabled = app_config.RACE_MODE_ENABLED and not is_auto_attempt
    race_count = app_config.RACE_CONCURRENT_COUNT if race_enabled else 1
    
    if race_enabled:
        print(f"INFO: Race mode ENABLED - will make {race_count} concurrent requests and use first successful one")
    
    if request_obj.stream:
        # For streaming, if race mode is enabled, we race the true streams
        if race_enabled:
            print("INFO: Race mode (streaming) - racing true streams to get the fastest first chunk.")
            
            async def create_generator_factory(client_factory_func):
                # This closure captures the client factory for the generator
                async def generator_factory():
                    stream_client = await client_factory_func()
                    stream_gen_obj = await stream_client.aio.models.generate_content_stream(
                        model=model_to_call, 
                        contents=actual_prompt_for_call,
                        config=gen_config_dict
                    )
                    response_id_for_stream = f"chatcmpl-racestream-{int(time.time())}"
                    async for chunk_item_call in stream_gen_obj:
                        yield convert_chunk_to_openai(chunk_item_call, request_obj.model, response_id_for_stream, 0)
                    yield "data: [DONE]\n\n"
                return generator_factory

            # Create a list of generator factories, one for each concurrent request
            generator_factories = [await create_generator_factory(client_factory) for _ in range(race_count)]
            
            return StreamingResponse(race_streaming_generators(generator_factories), media_type="text/event-stream")

        # Standard true streaming or fake streaming if enabled
        if app_config.FAKE_STREAMING_ENABLED:
            print("INFO: Fake streaming is enabled.")
            return StreamingResponse(
                gemini_fake_stream_generator(
                    client_factory, model_to_call, actual_prompt_for_call,
                    gen_config_dict, 
                    request_obj, is_auto_attempt, 1 # race_count is 1 here
                ), media_type="text/event-stream"
            )
        else: # True Streaming (no race mode)
            response_id_for_stream = f"chatcmpl-realstream-{int(time.time())}"
            async def _gemini_real_stream_generator_inner():
                try:
                    # For true streaming, we only create one client.
                    stream_client = await client_factory()
                    stream_gen_obj = await stream_client.aio.models.generate_content_stream(
                        model=model_to_call, 
                        contents=actual_prompt_for_call,
                        config=gen_config_dict
                    )
                    async for chunk_item_call in stream_gen_obj:
                        yield convert_chunk_to_openai(chunk_item_call, request_obj.model, response_id_for_stream, 0)
                    yield "data: [DONE]\n\n"
                except Exception as e_stream_call:
                    err_msg_detail_stream = f"Streaming Error (Gemini API, model string: '{model_to_call}'): {type(e_stream_call).__name__} - {str(e_stream_call)}"
                    print(f"ERROR: {err_msg_detail_stream}")
                    s_err = str(e_stream_call); s_err = s_err[:1024]+"..." if len(s_err)>1024 else s_err
                    err_resp = create_openai_error_response(500,s_err,"server_error")
                    j_err = json.dumps(err_resp)
                    if not is_auto_attempt: 
                        yield f"data: {j_err}\n\n"
                        yield "data: [DONE]\n\n"
                    raise e_stream_call
            return StreamingResponse(_gemini_real_stream_generator_inner(), media_type="text/event-stream")
    else: # Non-streaming
        async def _single_gemini_call():
            # Each call to the factory gets a new client, perfect for key rotation in race mode.
            gemini_client = await client_factory()
            response_obj_call = await gemini_client.aio.models.generate_content(
                model=model_to_call, 
                contents=actual_prompt_for_call,
                config=gen_config_dict
            )
            if hasattr(response_obj_call, 'prompt_feedback') and \
               hasattr(response_obj_call.prompt_feedback, 'block_reason') and \
               response_obj_call.prompt_feedback.block_reason:
                block_msg = f"Blocked (Gemini): {response_obj_call.prompt_feedback.block_reason}"
                if hasattr(response_obj_call.prompt_feedback,'block_reason_message') and \
                   response_obj_call.prompt_feedback.block_reason_message: 
                    block_msg+=f" ({response_obj_call.prompt_feedback.block_reason_message})"
                raise ValueError(block_msg)
            
            if not is_gemini_response_valid(response_obj_call):
                error_details = f"Invalid non-streaming Gemini response for model string '{model_to_call}'. Response: {response_obj_call}"
                raise ValueError(error_details)
            
            return response_obj_call
        
        # Use race mode for non-streaming if enabled
        if race_enabled:
            print("INFO: Race mode (non-streaming) - waiting for all requests to complete to select the longest response.")
            response_obj_call = await race_for_longest_string(_single_gemini_call, race_count)
        else:
            response_obj_call = await _single_gemini_call()
        
        openai_response_content = convert_to_openai_format(response_obj_call, request_obj.model)
        return JSONResponse(content=openai_response_content)