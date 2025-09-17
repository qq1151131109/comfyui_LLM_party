import asyncio
import base64
import datetime
import gc
import glob
import hashlib
import importlib
import io
import json
import locale
import os
import random
import re
import sys
import time
import traceback
# import google.generativeai as genai
import numpy as np
import openai
import aisuite as ai
import requests
import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM, 
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    AutoConfig,
)
import PIL.Image
from openai import AzureOpenAI, OpenAI, APIError, APITimeoutError, RateLimitError, AuthenticationError
if torch.cuda.is_available():
    from transformers import BitsAndBytesConfig
from google.protobuf.struct_pb2 import Struct
from torchvision.transforms import ToPILImage
from .config_update import models_dict
from .config import config_key, config_path, current_dir_path, load_api_keys
from .tools.lorebook import Lorebook
from .tools.api_tool import (
    api_function,
    api_tool,
    json2text,
    list_append,
    list_append_plus,
    list_extend,
    list_extend_plus,
    parameter_combine,
    parameter_combine_plus,
    parameter_function,
    use_api_tool,
)
from .tools.check_web import check_web, check_web_tool
from .tools.classify_function import classify_function, classify_function_plus
from .tools.classify_persona import classify_persona, classify_persona_plus
from .tools.clear_file import clear_file
from .tools.clear_model import clear_model
from .tools.custom_persona import custom_persona
from .tools.dialog import end_dialog, start_dialog,start_anything,end_anything
from .tools.dingding import Dingding, Dingding_tool, send_dingding
from .tools.end_work import end_workflow, img2path
from .tools.excel import image_iterator, load_excel,json_iterator,file_path_iterator
from .tools.feishu import feishu, feishu_tool, send_feishu
from .tools.file_combine import file_combine, file_combine_plus,string_combine, string_combine_plus
from .tools.get_time import get_time, time_tool
from .tools.get_weather import (
    accuweather_tool,
    get_accuweather,
)
from .tools.git_tool import github_tool, search_github_repositories
from .tools.image import CLIPTextEncode_party, KSampler_party, VAEDecode_party
from .tools.interpreter import interpreter, interpreter_function, interpreter_tool
from .tools.keyword import keyword_tool, load_keyword, search_keyword
from .tools.KG import (
    Delete_entities,
    Delete_relationships,
    Inquire_entities,
    Inquire_entity_list,
    Inquire_entity_relationships,
    Inquire_relationships,
    KG_json_toolkit_developer,
    KG_json_toolkit_user,
    Modify_entities,
    Modify_relationships,
    New_entities,
    New_relationships,
)
from .tools.KG_csv import (
    Delete_triple,
    Inquire_triple,
    KG_csv_toolkit_developer,
    KG_csv_toolkit_user,
    New_triple,
)
from .tools.KG_neo4j import (
    Delete_entities_neo4j,
    Delete_relationships_neo4j,
    Inquire_entities_neo4j,
    Inquire_entity_list_neo4j,
    Inquire_entity_relationships_neo4j,
    Inquire_relationships_neo4j,
    KG_neo_toolkit_developer,
    KG_neo_toolkit_user,
    Modify_entities_neo4j,
    Modify_relationships_neo4j,
    New_entities_neo4j,
    New_relationships_neo4j,
)
from .tools.load_ebd import data_base, ebd_tool, embeddings_function, save_ebd_database,load_ebd
from .tools.load_file import (
    load_file,
    load_file_folder,
    load_img_path,
    load_url,
    start_workflow,
)
from .tools.load_model_name import load_name
from .tools.load_persona import load_persona
from .tools.logic import get_string, replace_string, string_logic, substring
from .tools.omost import omost_decode, omost_setting
from .tools.search_web import (
    bing_loader,
    bing_tool,
    google_loader,
    google_tool,
    search_web,
    search_web_bing,
    duckduckgo_tool,
    duckduckgo_loader,
    search_duckduckgo,
)
from .tools.show_text import About_us, show_text_party
from .tools.smalltool import bool_logic, load_int, none2false,str2float,str2int,any2str,load_float,load_bool
from .tools.story import read_story_json, story_json_tool
from .tools.text_iterator import text_iterator,text_writing,json_writing
from .tools.tool_combine import tool_combine, tool_combine_plus
from .tools.translate_persona import translate_persona
from .tools.tts import openai_tts
from .tools.wechat import send_wechat, work_wechat, work_wechat_tool
from .tools.wikipedia import get_wikipedia, load_wikipedia, wikipedia_tool
from .tools.workflow import work_flow, workflow_tool, workflow_transfer
from .tools.flux_persona import flux_persona
from .tools.workflow_V2 import workflow_transfer_v2
os.environ["no_proxy"] = "localhost,127.0.0.1"
enable_interpreter = True

def retry_api_call(func, max_retries=3, base_delay=1):
    """
    指数退避重试机制
    对网络错误、超时错误进行重试，对API错误直接抛出
    """
    for attempt in range(max_retries):
        try:
            return func()
        except (APITimeoutError, ConnectionError, OSError) as e:
            if attempt == max_retries - 1:
                print(f"❌ 网络连接失败，已重试{max_retries}次: {str(e)}")
                raise e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"🔄 网络错误，{delay:.1f}秒后重试 (第{attempt+1}/{max_retries}次): {str(e)}")
            time.sleep(delay)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                print(f"❌ API调用频率限制，已重试{max_retries}次: {str(e)}")
                raise e
            delay = 60 + random.uniform(0, 30)  # API限制错误等待更久
            print(f"⏱️ API调用频率限制，{delay:.1f}秒后重试 (第{attempt+1}/{max_retries}次)")
            time.sleep(delay)
        except (APIError, AuthenticationError) as e:
            # API错误和认证错误一般不需要重试
            print(f"❌ API错误，不进行重试: {str(e)}")
            raise e
        except Exception as e:
            # 其他未知错误，记录并重试
            if attempt == max_retries - 1:
                print(f"❌ 未知错误，已重试{max_retries}次: {str(e)}")
                raise e
            delay = base_delay * (2 ** attempt)
            print(f"⚠️ 未知错误，{delay:.1f}秒后重试 (第{attempt+1}/{max_retries}次): {str(e)}")
            time.sleep(delay)

_TOOL_HOOKS = [
    "get_time",
    "search_web",
    "search_web_bing",
    "check_web",
    # "interpreter",
    "data_base",
    "another_llm",
    "use_api_tool",
    "get_accuweather",
    "get_wikipedia",
    "work_flow",
    "search_github_repositories",
    "send_wechat",
    "send_dingding",
    "send_feishu",
    "search_keyword",
    "read_story_json",
    "Inquire_entities",
    "New_entities",
    "Modify_entities",
    "Delete_entities",
    "Inquire_relationships",
    "New_relationships",
    "Modify_relationships",
    "Delete_relationships",
    "Inquire_triple",
    "New_triple",
    "Delete_triple",
    "Inquire_entity_relationships",
    "Inquire_entity_list",
    "Inquire_entities_neo4j",
    "New_entities_neo4j",
    "Modify_entities_neo4j",
    "Delete_entities_neo4j",
    "Inquire_relationships_neo4j",
    "New_relationships_neo4j",
    "Modify_relationships_neo4j",
    "Delete_relationships_neo4j",
    "Inquire_entity_relationships_neo4j",
    "Inquire_entity_list_neo4j",
    "search_duckduckgo",
]
if enable_interpreter:
    _TOOL_HOOKS.append("interpreter")
instances = []
image_buffer = []


def another_llm(id, type, question):
    print(id, type, question)
    global instances
    if type == "api":
        try:
            llm = next((instance for instance in instances if str(instance.id).strip() == str(id).strip()), None)
        except:
            print("找不到对应的智能助手")
            return "找不到对应的智能助手"
        if llm is None:
            print("找不到对应的智能助手")
            return "找不到对应的智能助手"
        (
            main_brain,
            system_prompt,
            model,
            temperature,
            is_memory,
            is_tools_in_sys_prompt,
            is_locked,
            max_length,
            system_prompt_input,
            user_prompt_input,
            tools,
            file_content,
            images,
            imgbb_api_key,
            conversation_rounds,
            historical_record,
            is_enable,
            extra_parameters,
            user_history,
            img_URL,
            stream,
        ) = llm.list
        res, _, _, _ = llm.chatbot(
            question,
            main_brain,
            system_prompt,
            model,
            temperature,
            is_memory,
            is_tools_in_sys_prompt,
            is_locked,
            max_length,
            system_prompt_input,
            user_prompt_input,
            tools,
            file_content,
            images,
            imgbb_api_key,
            conversation_rounds,
            historical_record,
            is_enable,
            extra_parameters,
            user_history,
            img_URL,
            stream,
        )
    elif type == "local":
        try:
            llm = next((instance for instance in instances if str(instance.id).strip() == str(id).strip()), None)
        except:
            print("找不到对应的智能助手")
            return "找不到对应的智能助手"
        if llm is None:
            print("找不到对应的智能助手")
            return "找不到对应的智能助手"
        (
            main_brain,
            system_prompt,
            model_type,
            model,
            tokenizer,
            temperature,
            max_length,
            is_memory,
            is_locked,
            system_prompt_input,
            user_prompt_input,
            tools,
            file_content,
            image,
            conversation_rounds,
            historical_record,
            is_enable,
            extra_parameters,
            user_history,
            is_enable_system_role,
        ) = llm.list
        res, _, _, _ = llm.chatbot(
            question,
            main_brain,
            system_prompt,
            model_type,
            model,
            tokenizer,
            temperature,
            max_length,
            is_memory,
            is_locked,
            system_prompt_input,
            user_prompt_input,
            tools,
            file_content,
            image,
            conversation_rounds,
            historical_record,
            is_enable,
            extra_parameters,
            user_history,
            is_enable_system_role,
        )
    else:
        return "type参数错误，请使用api或local"
    print(res)
    return "你调用的智能助手的问答是：" + res + "\n请根据以上回答，回答用户的问题。"


llm_tools_list = []
llm_tools = [
    {
        "type": "function",
        "function": {
            "name": "another_llm",
            "description": "使用llm_tools可以调用其他的智能助手解决你的问题。请根据以下列表中的system_prompt选择你需要的智能助手："
            + json.dumps(llm_tools_list, ensure_ascii=False, indent=4),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "智能助手的id"},
                    "type": {"type": "string", "description": "智能助手的类型，目前支持api和local两种类型。"},
                    "question": {"type": "string", "description": "问题描述，代表你想要解决的问题。"},
                },
                "required": ["id", "type", "question"],
            },
        },
    }
]


def dispatch_tool(tool_name: str, tool_params: dict) -> str:
    if "multi_tool_use." in tool_name:
        tool_name = tool_name.replace("multi_tool_use.", "")
    if '-' in tool_name and tool_name not in _TOOL_HOOKS:
        from .custom_tool.mcp_cli import mcp_client as client
        
        async def run_client():
            try:
                # Initialize the client (if necessary)
                if client.is_initialized is False:
                    await client.initialize()
                functions = await client.get_openai_functions()
                # Call the tool and get the result
                result = await client.call_tool(tool_name, tool_params)
                return str(result)
            except Exception as e:
                return str(e)

        # Run the async function using asyncio.run
        return asyncio.run(run_client())
    if tool_name not in _TOOL_HOOKS:
        return f"Tool `{tool_name}` not found. Please use a provided tool."
    tool_call = globals().get(tool_name)
    try:
        ret_out = tool_call(**tool_params)
        if isinstance(ret_out, tuple):
            ret = ret_out[0]
            global image_buffer
            image_buffer = ret_out[1]
            if ret == "" or ret is None:
                ret = "图片已生成。并以展示在本节点的image输出中。可以使用preview_image节点查看图片。"
        else:
            ret = ret_out
    except:
        ret = traceback.format_exc()
    return str(ret)

def ensure_version_suffix(base_url):
    """
    确保 base_url 以 '/v<n>/' 结尾，其中 n 是数字。
    如果不是这样，则添加 '/v1/' 到末尾。
    """
    # 正则表达式匹配 '/v<n>/' 其中 <n> 是任意字符
    match = re.search(r'/v(.+)/$', base_url)
    
    if not match:
        # 如果没有匹配到，则添加 '/v1/'
        if not base_url.endswith('/'):
            base_url += '/'
        base_url += 'v1/'
    return base_url

class Chat:
    def __init__(self, model_name, apikey, baseurl) -> None:
        self.model_name = model_name
        self.apikey = apikey
        self.baseurl = ensure_version_suffix(baseurl)

    def send(
        self,
        user_prompt,
        temperature,
        max_length,
        history,
        tools=None,
        is_tools_in_sys_prompt="disable",
        images=None,
        imgbb_api_key="",
        img_URL=None,
        stream=False,
        **extra_parameters,
    ):
        try:
            if images is not None and (img_URL is None or img_URL == ""):
                if imgbb_api_key == "" or imgbb_api_key is None:
                    imgbb_api_key = api_keys.get("imgbb_api")
                if imgbb_api_key == "" or imgbb_api_key is None:
                    img_json = [{"type": "text", "text": user_prompt}]
                    for image in images:
                        i = 255.0 * image.cpu().numpy()
                        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        img_json.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_str}"}
                        })
                    user_prompt = img_json
                else:
                    img_json = [{"type": "text", "text": user_prompt}]
                    for image in images:
                        i = 255.0 * image.cpu().numpy()
                        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        url = "https://api.imgbb.com/1/upload"
                        payload = {"key": imgbb_api_key, "image": img_str}
                        response = requests.post(url, data=payload)
                        if response.status_code == 200:
                            result = response.json()
                            img_url = result["data"]["url"]
                            img_json.append({
                                "type": "image_url",
                                "image_url": {"url": img_url}
                            })
                        else:
                            return "Error: " + response.text
                    user_prompt = img_json
            elif img_URL is not None and img_URL != "":
                user_prompt = [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": img_URL}}]
            # 将history中的系统提示词部分如果为空，就剔除
            for i in range(len(history)):
                if history[i]["role"] == "system" and history[i]["content"] == "":
                    history.pop(i)
                    break
            if re.search(r'o[1-3]', self.model_name):
                # 将history中的系统提示词部分的角色换成user
                for i in range(len(history)):
                    if history[i]["role"] == "system":
                        history[i]["role"] = "user"
                        history.append({"role": "assistant", "content": "好的，我会按照你的指示来操作"})
                        break
            openai_client = OpenAI(
                    api_key= self.apikey,
                    base_url=self.baseurl,
                    timeout=30.0,  # 30秒超时
                    max_retries=0,  # 禁用内置重试，使用自定义重试
                )
            if "openai.azure.com" in self.baseurl:
                # 获取API版本
                api_version = self.baseurl.split("=")[-1].split("/")[0]
                # 获取azure_endpoint
                azure_endpoint = "https://"+self.baseurl.split("//")[1].split("/")[0]
                azure = AzureOpenAI(
                    api_key= self.apikey,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    timeout=30.0,  # 30秒超时
                    max_retries=0,  # 禁用内置重试，使用自定义重试
                )
                openai_client = azure
            new_message = {"role": "user", "content": user_prompt}
            history.append(new_message)
            print(history)
            reasoning_content = ""
            if tools is not None:
                def api_call_with_tools():
                    return openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=history,
                        temperature=temperature,
                        tools=tools,
                        max_tokens=max_length,
                        stream=stream,
                        **extra_parameters,
                    )
                response = retry_api_call(api_call_with_tools)
                if stream:
                    tool_calls = []
                    response_content = ""
                    reasoning_content = ""
                    for chunk in response:
                        if chunk.choices:
                            choice = chunk.choices[0]
                            if choice.delta.tool_calls:  # function_calling
                                for idx, tool_call in enumerate(choice.delta.tool_calls):
                                    tool = choice.delta.tool_calls[idx]
                                    if len(tool_calls) <= idx:
                                        tool_calls.append(tool)
                                        continue
                                    if tool.function.arguments:
                                        # function参数为流式响应，需要拼接
                                        tool_calls[idx].function.arguments += tool.function.arguments
                            else: 
                                if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                                    print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
                                    reasoning_content = reasoning_content + chunk.choices[0].delta.reasoning_content
                                elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                                    print(chunk.choices[0].delta.content, end="", flush=True)
                                    response_content = response_content + chunk.choices[0].delta.content
                    while tool_calls:
                        response_content = tool_calls[0].function
                        print("正在调用" + response_content.name + "工具")
                        print(response_content.arguments)
                        results = dispatch_tool(response_content.name, json.loads(response_content.arguments))
                        print(results)
                        history.append(
                            {
                                "tool_calls": [
                                    {
                                        "id": tool_calls[0].id,
                                        "function": {
                                            "arguments": response_content.arguments,
                                            "name": response_content.name,
                                        },
                                        "type": tool_calls[0].type,
                                    }
                                ],
                                "role": "assistant",
                                "content": str(response_content),
                            }
                        )
                        history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_calls[0].id,
                                "name": response_content.name,
                                "content": results,
                            }
                        )
                        def api_call():

                            return openai_client.chat.completions.create(

                            model=self.model_name,
                            messages=history,
                            tools=tools,
                            temperature=temperature,
                            max_tokens=max_length,
                            stream=stream,
                            **extra_parameters,
                        

                            )

                        response = retry_api_call(api_call)
                        tool_calls = []
                        response_content = ""
                        reasoning_content = ""
                        for chunk in response:
                            if chunk.choices:
                                choice = chunk.choices[0]
                                if choice.delta.tool_calls:  # function_calling
                                    for idx, tool_call in enumerate(choice.delta.tool_calls):
                                        tool = choice.delta.tool_calls[idx]
                                        if len(tool_calls) <= idx:
                                            tool_calls.append(tool)
                                            continue
                                        if tool.function.arguments:
                                            # function参数为流式响应，需要拼接
                                            tool_calls[idx].function.arguments += tool.function.arguments
                                else:                          
                                    if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                                        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
                                        reasoning_content = reasoning_content + chunk.choices[0].delta.reasoning_content
                                    elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                                        print(chunk.choices[0].delta.content, end="", flush=True)
                                        response_content = response_content + chunk.choices[0].delta.content
                else:
                    response_content = response.choices[0].message.content
                    print(response_content)
                    while response.choices[0].message.tool_calls:
                        assistant_message = response.choices[0].message
                        response_content = assistant_message.tool_calls[0].function
                        print("正在调用" + response_content.name + "工具")
                        print(response_content.arguments)
                        results = dispatch_tool(response_content.name, json.loads(response_content.arguments))
                        print(results)
                        history.append(
                            {
                                "tool_calls": [
                                    {
                                        "id": assistant_message.tool_calls[0].id,
                                        "function": {
                                            "arguments": response_content.arguments,
                                            "name": response_content.name,
                                        },
                                        "type": assistant_message.tool_calls[0].type,
                                    }
                                ],
                                "role": "assistant",
                                "content": str(response_content),
                            }
                        )
                        history.append(
                            {
                                "role": "tool",
                                "tool_call_id": assistant_message.tool_calls[0].id,
                                "name": response_content.name,
                                "content": results,
                            }
                        )
                        def api_call():

                            return openai_client.chat.completions.create(

                            model=self.model_name,
                            messages=history,
                            tools=tools,
                            temperature=temperature,
                            max_tokens=max_length,
                            **extra_parameters,
                        

                            )

                        response = retry_api_call(api_call)
                        if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
                            reasoning_content = response.choices[0].message.reasoning_content
                            print(reasoning_content)
                        response_content = response.choices[0].message.content
                        print(response_content)
            elif is_tools_in_sys_prompt == "enable":
                def api_call():

                    return openai_client.chat.completions.create(

                    model=self.model_name,
                    messages=history,
                    temperature=temperature,
                    max_tokens=max_length,
                    **extra_parameters,
                

                    )

                response = retry_api_call(api_call)
                response_content = response.choices[0].message.content
                # 正则表达式匹配
                pattern = r'\{\s*"tool":\s*"(.*?)",\s*"parameters":\s*\{(.*?)\}\s*\}'
                while re.search(pattern, response_content, re.DOTALL) != None:
                    match = re.search(pattern, response_content, re.DOTALL)
                    tool = match.group(1)
                    parameters = match.group(2)
                    json_str = '{"tool": "' + tool + '", "parameters": {' + parameters + "}}"
                    print("正在调用" + tool + "工具")
                    parameters = json.loads("{" + parameters + "}")
                    results = dispatch_tool(tool, parameters)
                    print(results)
                    history.append({"role": "assistant", "content": json_str})
                    history.append(
                        {
                            "role": "user",
                            "content": "调用"
                            + tool
                            + "工具返回的结果为："
                            + results
                            + "。请根据工具返回的结果继续回答我之前提出的问题。",
                        }
                    )
                    def api_call():

                        return openai_client.chat.completions.create(

                        model=self.model_name,
                        messages=history,
                        temperature=temperature,
                        max_tokens=max_length,
                        **extra_parameters,
                    

                        )

                    response = retry_api_call(api_call)
                    if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
                        reasoning_content = response.choices[0].message.reasoning_content
                        print(reasoning_content)
                    response_content = response.choices[0].message.content
                    print(response_content)
            else:
                def api_call():

                    return openai_client.chat.completions.create(

                    model=self.model_name,
                    messages=history,
                    temperature=temperature,
                    max_tokens=max_length,
                    stream=stream,
                    **extra_parameters,
                

                    )

                response = retry_api_call(api_call)
                response_content = ""
                reasoning_content = ""
                if stream:
                    for chunk in response:
                        if chunk.choices:
                            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                                print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
                                reasoning_content = reasoning_content + chunk.choices[0].delta.reasoning_content
                            elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                                print(chunk.choices[0].delta.content, end="", flush=True)
                                response_content = response_content + chunk.choices[0].delta.content
                    print(response_content)
                else:
                    if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
                        reasoning_content = response.choices[0].message.reasoning_content
                        print(reasoning_content)
                    response_content = response.choices[0].message.content
                    print(response_content)
            # 使用正则表达式，提取response_content中<think>到</think>之间的内容到reasoning_content，之外的内容到response_content
            pattern = r'<think>(.*?)</think>'
            match = re.search(pattern, response_content, re.DOTALL)
            if match:
                reasoning_content = match.group(1).strip()
                response_content = response_content.replace(match.group(0), "").strip() 
            history.append({"role": "assistant", "content": response_content})
        except Exception as ex:
            response_content = str(ex)
            reasoning_content = str(ex)
        return response_content, history,reasoning_content



class aisuite_Chat:
    def __init__(self, provider,model_name, apikey, baseurl,aws_access_key_id,aws_secret_access_key, aws_region_name,google_project_id,google_region,google_application_credentials,hf_api_token) -> None:
        self.model_name = f"{provider}:{model_name}"
        self.apikey = apikey
        self.baseurl = baseurl
        self.provider_configs = {}
        if provider == "openai":
            self.provider_configs["openai"] ={
                    "api_key": apikey,
                    "base_url": baseurl if baseurl != "" else "https://api.openai.com/v1"
                }
        elif provider == "anthropic":
            self.provider_configs["anthropic"] ={
                    "api_key": apikey,
                    "base_url": baseurl if baseurl != "" else "https://api.anthropic.com/v1"
                }
        elif provider == "azure":
            self.provider_configs["azure"] ={
                    "api_key": apikey,
                    "base_url": baseurl if baseurl != "" else "https://openai.azure.com/v1"
                }
        elif provider == "aws":
            os.environ['AWS_ACCESS_KEY'] = aws_access_key_id
            os.environ['AWS_SECRET_KEY'] = aws_secret_access_key
            os.environ['AWS_REGION_NAME'] = aws_region_name
        elif provider == "google":
            os.environ['GOOGLE_PROJECT_ID'] = google_project_id
            os.environ['GOOGLE_REGION'] = google_region
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =  google_application_credentials
        elif provider == "huggingface":
            os.environ['HUGGINGFACE_API_TOKEN'] = hf_api_token

    def send(
        self,
        user_prompt,
        temperature,
        max_length,
        history,
        tools=None,
        is_tools_in_sys_prompt="disable",
        images=None,
        imgbb_api_key="",
        img_URL=None,
        stream=False,
        **extra_parameters,
    ):
        try:
            if images is not None and (img_URL is None or img_URL == ""):
                if imgbb_api_key == "" or imgbb_api_key is None:
                    imgbb_api_key = api_keys.get("imgbb_api")
                if imgbb_api_key == "" or imgbb_api_key is None:
                    img_json = [{"type": "text", "text": user_prompt}]
                    for image in images:
                        i = 255.0 * image.cpu().numpy()
                        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        img_json.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_str}"}
                        })
                    user_prompt = img_json
                else:
                    img_json = [{"type": "text", "text": user_prompt}]
                    for image in images:
                        i = 255.0 * image.cpu().numpy()
                        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        url = "https://api.imgbb.com/1/upload"
                        payload = {"key": imgbb_api_key, "image": img_str}
                        response = requests.post(url, data=payload)
                        if response.status_code == 200:
                            result = response.json()
                            img_url = result["data"]["url"]
                            img_json.append({
                                "type": "image_url",
                                "image_url": {"url": img_url}
                            })
                        else:
                            return "Error: " + response.text
                    user_prompt = img_json
            elif img_URL is not None and img_URL != "":
                user_prompt = [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": img_URL}}]
            # 将history中的系统提示词部分如果为空，就剔除
            for i in range(len(history)):
                if history[i]["role"] == "system" and history[i]["content"] == "":
                    history.pop(i)
                    break
            if re.search(r'o[1-3]', self.model_name):
                # 将history中的系统提示词部分的角色换成user
                for i in range(len(history)):
                    if history[i]["role"] == "system":
                        history[i]["role"] = "user"
                        history.append({"role": "assistant", "content": "好的，我会按照你的指示来操作"})
                        break
            openai_client = ai.Client(self.provider_configs)
            if images is not None:
                if imgbb_api_key == "" or imgbb_api_key is None:
                    imgbb_api_key = api_keys.get("imgbb_api")
                if imgbb_api_key == "" or imgbb_api_key is None:
                    img_json = [{"type": "text", "text": user_prompt}]
                    for image in images:
                        i = 255.0 * image.cpu().numpy()
                        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        img_json.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_str}"}
                        })
                    user_prompt = img_json
                else:
                    img_json = [{"type": "text", "text": user_prompt}]
                    for image in images:
                        i = 255.0 * image.cpu().numpy()
                        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        url = "https://api.imgbb.com/1/upload"
                        payload = {"key": imgbb_api_key, "image": img_str}
                        response = requests.post(url, data=payload)
                        if response.status_code == 200:
                            result = response.json()
                            img_url = result["data"]["url"]
                            img_json.append({
                                "type": "image_url",
                                "image_url": {"url": img_url}
                            })
                        else:
                            return "Error: " + response.text
                    user_prompt = img_json

            # 将history中的系统提示词部分如果为空，就剔除
            for i in range(len(history)):
                if history[i]["role"] == "system" and history[i]["content"] == "":
                    history.pop(i)
                    break
            new_message = {"role": "user", "content": user_prompt}
            history.append(new_message)
            print(history)
            reasoning_content = ""
            if tools is not None:
                def api_call():

                    return openai_client.chat.completions.create(

                    model=self.model_name,
                    messages=history,
                    temperature=temperature,
                    tools=tools,
                    max_tokens=max_length,
                    stream=stream,
                    **extra_parameters,
                

                    )

                response = retry_api_call(api_call)
                if stream:
                    tool_calls = []
                    response_content = ""
                    reasoning_content = ""
                    for chunk in response:
                        if chunk.choices:
                            choice = chunk.choices[0]
                            if choice.delta.tool_calls:  # function_calling
                                for idx, tool_call in enumerate(choice.delta.tool_calls):
                                    tool = choice.delta.tool_calls[idx]
                                    if len(tool_calls) <= idx:
                                        tool_calls.append(tool)
                                        continue
                                    if tool.function.arguments:
                                        # function参数为流式响应，需要拼接
                                        tool_calls[idx].function.arguments += tool.function.arguments
                            else:       
                                if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                                    print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
                                    reasoning_content = reasoning_content + chunk.choices[0].delta.reasoning_content
                                elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                                    print(chunk.choices[0].delta.content, end="", flush=True)
                                    response_content = response_content + chunk.choices[0].delta.content
                    while tool_calls:
                        response_content = tool_calls[0].function
                        print("正在调用" + response_content.name + "工具")
                        print(response_content.arguments)
                        results = dispatch_tool(response_content.name, json.loads(response_content.arguments))
                        print(results)
                        history.append(
                            {
                                "tool_calls": [
                                    {
                                        "id": tool_calls[0].id,
                                        "function": {
                                            "arguments": response_content.arguments,
                                            "name": response_content.name,
                                        },
                                        "type": tool_calls[0].type,
                                    }
                                ],
                                "role": "assistant",
                                "content": str(response_content),
                            }
                        )
                        history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_calls[0].id,
                                "name": response_content.name,
                                "content": results,
                            }
                        )
                        def api_call():

                            return openai_client.chat.completions.create(

                            model=self.model_name,
                            messages=history,
                            tools=tools,
                            temperature=temperature,
                            max_tokens=max_length,
                            stream=stream,
                            **extra_parameters,
                        

                            )

                        response = retry_api_call(api_call)
                        tool_calls = []
                        response_content = ""
                        reasoning_content = ""
                        for chunk in response:
                            if chunk.choices:
                                choice = chunk.choices[0]
                                if choice.delta.tool_calls:  # function_calling
                                    for idx, tool_call in enumerate(choice.delta.tool_calls):
                                        tool = choice.delta.tool_calls[idx]
                                        if len(tool_calls) <= idx:
                                            tool_calls.append(tool)
                                            continue
                                        if tool.function.arguments:
                                            # function参数为流式响应，需要拼接
                                            tool_calls[idx].function.arguments += tool.function.arguments
                                else:                            
                                    if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                                        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
                                        reasoning_content = reasoning_content + chunk.choices[0].delta.reasoning_content
                                    elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                                        print(chunk.choices[0].delta.content, end="", flush=True)
                                        response_content = response_content + chunk.choices[0].delta.content
                else:
                    response_content = response.choices[0].message.content
                    print(response_content)
                    while response.choices[0].message.tool_calls:
                        assistant_message = response.choices[0].message
                        response_content = assistant_message.tool_calls[0].function
                        print("正在调用" + response_content.name + "工具")
                        print(response_content.arguments)
                        results = dispatch_tool(response_content.name, json.loads(response_content.arguments))
                        print(results)
                        history.append(
                            {
                                "tool_calls": [
                                    {
                                        "id": assistant_message.tool_calls[0].id,
                                        "function": {
                                            "arguments": response_content.arguments,
                                            "name": response_content.name,
                                        },
                                        "type": assistant_message.tool_calls[0].type,
                                    }
                                ],
                                "role": "assistant",
                                "content": str(response_content),
                            }
                        )
                        history.append(
                            {
                                "role": "tool",
                                "tool_call_id": assistant_message.tool_calls[0].id,
                                "name": response_content.name,
                                "content": results,
                            }
                        )
                        def api_call():

                            return openai_client.chat.completions.create(

                            model=self.model_name,
                            messages=history,
                            tools=tools,
                            temperature=temperature,
                            max_tokens=max_length,
                            **extra_parameters,
                        

                            )

                        response = retry_api_call(api_call)
                        if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
                            reasoning_content = response.choices[0].message.reasoning_content
                            print(reasoning_content)
                        response_content = response.choices[0].message.content
                        print(response_content)
            elif is_tools_in_sys_prompt == "enable":
                def api_call():

                    return openai_client.chat.completions.create(

                    model=self.model_name,
                    messages=history,
                    temperature=temperature,
                    max_tokens=max_length,
                    **extra_parameters,
                

                    )

                response = retry_api_call(api_call)
                response_content = response.choices[0].message.content
                # 正则表达式匹配
                pattern = r'\{\s*"tool":\s*"(.*?)",\s*"parameters":\s*\{(.*?)\}\s*\}'
                while re.search(pattern, response_content, re.DOTALL) != None:
                    match = re.search(pattern, response_content, re.DOTALL)
                    tool = match.group(1)
                    parameters = match.group(2)
                    json_str = '{"tool": "' + tool + '", "parameters": {' + parameters + "}}"
                    print("正在调用" + tool + "工具")
                    parameters = json.loads("{" + parameters + "}")
                    results = dispatch_tool(tool, parameters)
                    print(results)
                    history.append({"role": "assistant", "content": json_str})
                    history.append(
                        {
                            "role": "user",
                            "content": "调用"
                            + tool
                            + "工具返回的结果为："
                            + results
                            + "。请根据工具返回的结果继续回答我之前提出的问题。",
                        }
                    )
                    def api_call():

                        return openai_client.chat.completions.create(

                        model=self.model_name,
                        messages=history,
                        temperature=temperature,
                        max_tokens=max_length,
                        **extra_parameters,
                    

                        )

                    response = retry_api_call(api_call)
                    if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
                        reasoning_content = response.choices[0].message.reasoning_content
                        print(reasoning_content)
                    response_content = response.choices[0].message.content
                    print(response_content)
            else:
                def api_call():

                    return openai_client.chat.completions.create(

                    model=self.model_name,
                    messages=history,
                    temperature=temperature,
                    max_tokens=max_length,
                    stream=stream,
                    **extra_parameters,
                

                    )

                response = retry_api_call(api_call)
                response_content = ""
                reasoning_content = ""
                if stream:
                    for chunk in response:
                        if chunk.choices:
                            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                                print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
                                reasoning_content = reasoning_content + chunk.choices[0].delta.reasoning_content
                            elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                                print(chunk.choices[0].delta.content, end="", flush=True)
                                response_content = response_content + chunk.choices[0].delta.content
                else:
                    if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
                        reasoning_content = response.choices[0].message.reasoning_content
                        print(reasoning_content)
                    response_content = response.choices[0].message.content
                    print(response_content)
            # 使用正则表达式，提取response_content中<think>到</think>之间的内容到reasoning_content，之外的内容到response_content
            pattern = r'<think>(.*?)</think>'
            match = re.search(pattern, response_content, re.DOTALL)
            if match:
                reasoning_content = match.group(1).strip()
                response_content = response_content.replace(match.group(0), "").strip() 
            history.append({"role": "assistant", "content": response_content})
        except Exception as ex:
            response_content = str(ex)
            reasoning_content = str(ex)
        return response_content, history,reasoning_content



class LLM_api_loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ("STRING", {"default": "gpt-4o-mini","tooltip": "The name of the model, such as gpt-4o-mini."}),
            },
            "optional": {
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The base URL of the API, such as https://api.openai.com/v1.",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The API key for the API."
                    },
                ),
                "is_ollama": ("BOOLEAN", {"default": False, "tooltip": "Whether to use ollama."}),
            },
        }

    RETURN_TYPES = ("CUSTOM",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("The loaded model.",)
    DESCRIPTION = "Load the model in openai format."
    FUNCTION = "chatbot"

    # OUTPUT_NODE = False

    CATEGORY = "大模型派对（llm_party）/模型加载器（model loader）"

    def chatbot(self, model_name, base_url=None, api_key=None, is_ollama=False):
        if is_ollama:
            openai.api_key = "ollama"
            openai.base_url = "http://127.0.0.1:11434/v1/"
        else:
            api_keys = load_api_keys(config_path)
            if api_key != "":
                openai.api_key = api_key
            elif model_name in config_key:
                api_keys = config_key[model_name]
                openai.api_key = api_keys.get("api_key")
            elif api_keys.get("openai_api_key") != "":
                openai.api_key = api_keys.get("openai_api_key")
            if base_url != "":
                openai.base_url = base_url
            elif model_name in config_key:
                api_keys = config_key[model_name]
                openai.base_url = api_keys.get("base_url")
            elif api_keys.get("base_url") != "":
                openai.base_url = api_keys.get("base_url")
            if openai.api_key == "":
                api_keys = load_api_keys(config_path)
                openai.api_key = api_keys.get("openai_api_key")
                openai.base_url = api_keys.get("base_url")
            if openai.base_url != "":
                if openai.base_url[-1] != "/":
                    openai.base_url = openai.base_url + "/"

        chat = Chat(model_name, openai.api_key, openai.base_url)
        return (chat,)
    
class aisuite_loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["openai","anthropic","aws","azure","vertex","huggingface"], {"default": "openai","tooltip": "API interface type"}),
                "model_name": ("STRING", {"default": "gpt-4o-mini", "tooltip": "The name of the model, such as gpt-4o-mini."}),
            },
            "optional": {
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The base URL of the API, such as https://api.openai.com/v1.",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The API key for the API."
                    },
                ),
                "aws_access_key_id":(
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The AWS access key ID."
                    }
                ),
                "aws_secret_access_key":(
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The AWS secret access key."
                    }
                ),
                "aws_region_name":(
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The AWS region name."
                    }
                ),
                "google_project_id":(
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The Google project ID."
                    }
                ),
                "google_region":(
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The Google region."
                    }
                ),
                "google_application_credentials":(
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The Google application credentials."
                    }
                ),
                "hf_api_token":(
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The Hugging Face API token."
                    }
                ),
            },
        }

    RETURN_TYPES = ("CUSTOM",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("The loaded model.",)
    DESCRIPTION = "Load the model in aisuite format."
    FUNCTION = "chatbot"

    # OUTPUT_NODE = False

    CATEGORY = "大模型派对（llm_party）/模型加载器（model loader）"

    def chatbot(self, provider,model_name,aws_access_key_id,aws_secret_access_key, aws_region_name,google_project_id,google_region,google_application_credentials,hf_api_token, base_url=None, api_key=None):
        api_keys = load_api_keys(config_path)
        if api_key != "":
            openai.api_key = api_key
        elif model_name in config_key:
            api_keys = config_key[model_name]
            openai.api_key = api_keys.get("api_key")
        elif api_keys.get("openai_api_key") != "":
            openai.api_key = api_keys.get("openai_api_key")
        if base_url != "":
            openai.base_url = base_url
        elif model_name in config_key:
            api_keys = config_key[model_name]
            openai.base_url = api_keys.get("base_url")
        elif api_keys.get("base_url") != "":
            openai.base_url = api_keys.get("base_url")
        if openai.api_key == "":
            api_keys = load_api_keys(config_path)
            openai.api_key = api_keys.get("openai_api_key")
            openai.base_url = api_keys.get("base_url")
        if openai.base_url != "":
            if openai.base_url[-1] != "/":
                openai.base_url = openai.base_url + "/"

        chat = aisuite_Chat(provider,model_name, openai.api_key, openai.base_url,aws_access_key_id,aws_secret_access_key, aws_region_name,google_project_id,google_region,google_application_credentials,hf_api_token)
        return (chat,)


class easy_LLM_api_loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (models_dict, {"default": "", "tooltip": "The model name."}),
            },
        }

    RETURN_TYPES = ("CUSTOM",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("The loaded model.",)
    FUNCTION = "chatbot"
    DESCRIPTION = "Load a model from the Easy mode."
    # OUTPUT_NODE = False

    CATEGORY = "大模型派对（llm_party）/模型加载器（model loader）"

    def chatbot(self, model_name):
        # 如果openai.base_url没有/结尾就加一个
        if openai.base_url != "":
            if openai.base_url[-1] != "/":
                openai.base_url = openai.base_url + "/"
        chat = Chat(model_name, openai.api_key, openai.base_url)
        return (chat,)

class LLM:
    def __init__(self):
        current_time = datetime.datetime.now()
        # 以时间戳作为ID，字符串格式 XX年XX月XX日XX时XX分XX秒并加上一个哈希值防止重复
        self.id = current_time.strftime("%Y_%m_%d_%H_%M_%S") + str(hash(random.randint(0, 1000000)))
        global instances
        instances.append(self)
        # 构建prompt.json的绝对路径，如果temp文件夹不存在就创建
        current_dir_path = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(current_dir_path, "temp"), exist_ok=True)
        self.prompt_path = os.path.join(current_dir_path, "temp", str(self.id) + ".json")
        # 如果文件不存在，创建prompt.json文件，存在就覆盖文件
        if not os.path.exists(self.prompt_path):
            with open(self.prompt_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{"role": "system", "content": "你是一个强大的人工智能助手。"}], f, indent=4, ensure_ascii=False
                )
        self.tool_data = {"id": self.id, "system_prompt": "", "type": "api"}
        self.list = []
        self.added_to_list = False
        self.is_locked = "disable"

    @classmethod
    def INPUT_TYPES(s):
        temp_path = os.path.join(current_dir_path, "temp")
        full_paths = [os.path.join(temp_path, f) for f in os.listdir(temp_path)]
        full_paths.sort(key=os.path.getmtime, reverse=True)
        paths = [os.path.basename(f) for f in full_paths]
        paths.insert(0, "")
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": True, "default": "你一个强大的人工智能助手。","tooltip": "System prompt, used to describe the behavior of the model and the expected output format."}),
                "user_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "你好",
                        "tooltip": "User prompt, used to describe the user's request and the expected output format."
                    },
                ),
                "model": ("CUSTOM", {"tooltip": "The model to use for the LLM."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1,"tooltip": "The temperature parameter controls the randomness of the model's output. A higher temperature will result in more random and diverse responses, while a lower temperature will result in more focused and deterministic responses."}),
                "is_memory": (["enable", "disable"], {"default": "enable", "tooltip": "Whether to enable memory for the LLM."}),
                "is_tools_in_sys_prompt": (["enable", "disable"], {"default": "disable", "tooltip": "Integrate the tool list into the system prompt, thereby granting some models temporary capability to invoke tools."}),
                "is_locked": (["enable", "disable"], {"default": "disable", "tooltip": "Whether to directly output the result from the last output."}),
                "main_brain": (["enable", "disable"], {"default": "enable", "tooltip": "If this option is disabled, the LLM will become a tool that can be invoked by other LLMs."}),
                "max_length": ("INT", {"default": 1920, "min": 256, "max": 128000, "step": 128, "tooltip": "The maximum length of the output text."}),
            },
            "optional": {
                "system_prompt_input": ("STRING", {"forceInput": True, "tooltip": "System prompt input, used to describe the system's request and the expected output format."}),
                "user_prompt_input": ("STRING", {"forceInput": True, "tooltip": "User prompt input, used to describe the user's request and the expected output format."}),
                "tools": ("STRING", {"forceInput": True, "tooltip": "Tool list, used to describe the tools that the model can invoke."}),
                "file_content": ("STRING", {"forceInput": True, "tooltip": "Input the contents of the file here."}),
                "images": ("IMAGE", {"forceInput": True, "tooltip": "Upload images here."}),
                "imgbb_api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional, if not filled out, it will be passed to the LLM in the form of a base64 encoded string. API key for ImgBB, used to upload images to ImgBB and get the image URL."
                    },
                ),
                "conversation_rounds": ("INT", {"default": 100, "min": 1, "max": 10000, "step": 1, "tooltip": "The maximum number of dialogue turns that the LLM can see in the history records, where one question and one answer constitute one turn."}),
                "historical_record": (paths, {"default": "", "tooltip": "The dialogue history file is optional; if not selected and left empty, a new dialogue history file will be automatically created."}),
                "is_enable": ("BOOLEAN", {"default": True, "tooltip": "Whether to enable the LLM."}),
                "extra_parameters": ("DICT", {"forceInput": True, "tooltip": "Extra parameters for the LLM."}),
                "user_history": ("STRING", {"forceInput": True, "tooltip": "User history, you can directly input a JSON string containing multiple rounds of dialogue here."}),
                "img_URL": ("STRING", {"forceInput": True, "tooltip": "The URL of the image."}),
                "stream": ("BOOLEAN", {"default": False, "tooltip": "Whether to enable streaming output."}),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "IMAGE",
        "STRING",
    )
    RETURN_NAMES = (
        "assistant_response",
        "history",
        "tool",
        "image",
        "reasoning_content",
    )
    OUTPUT_TOOLTIPS = (
        "The assistant's response to the user's request.",
        "The dialogue history",
        "This interface will connect this LLM as a tool to other LLMs.",
        "Images generated or fetched by the LLM.",
        "The assistant's reasoning process."
    )
    DESCRIPTION = "The API version of the model chain, compatible with all API interfaces."
    FUNCTION = "chatbot"

    # OUTPUT_NODE = False

    CATEGORY = "大模型派对（llm_party）/模型链（model_chain）"

    def chatbot(
        self,
        user_prompt,
        main_brain,
        system_prompt,
        model,
        temperature,
        is_memory,
        is_tools_in_sys_prompt,
        is_locked,
        max_length,
        system_prompt_input="",
        user_prompt_input="",
        tools=None,
        file_content=None,
        images=None,
        imgbb_api_key=None,
        conversation_rounds=100,
        historical_record="",
        is_enable=True,
        extra_parameters=None,
        user_history=None,
        img_URL=None,
        stream=False,
    ):
        if not is_enable:
            return (
                None,
                None,
                None,
                None,
                "",
            )
        self.list = [
            main_brain,
            system_prompt,
            model,
            temperature,
            is_memory,
            is_tools_in_sys_prompt,
            is_locked,
            max_length,
            system_prompt_input,
            user_prompt_input,
            tools,
            file_content,
            images,
            imgbb_api_key,
            conversation_rounds,
            historical_record,
            is_enable,
            extra_parameters,
            user_history,
            img_URL,
            stream,
        ]
        if user_prompt is None:
            user_prompt = user_prompt_input
        elif user_prompt_input is not None:
            user_prompt = user_prompt + user_prompt_input
        if historical_record != "":
            temp_path = os.path.join(current_dir_path, "temp")
            self.prompt_path = os.path.join(temp_path, historical_record)
        self.tool_data["system_prompt"] = system_prompt
        if system_prompt_input is not None and system_prompt is not None:
            system_prompt = system_prompt + system_prompt_input
        elif system_prompt is None:
            system_prompt = system_prompt_input
        global llm_tools_list, llm_tools
        if main_brain == "disable":
            if self.added_to_list == False:
                llm_tools_list.append(self.tool_data)
                self.added_to_list = True
        self.is_locked = is_locked
        if self.is_locked == "disable":
            setattr(LLM, "IS_CHANGED", LLM.original_IS_CHANGED)
        else:
            # 如果方法存在，则删除
            if hasattr(LLM, "IS_CHANGED"):
                delattr(LLM, "IS_CHANGED")
        llm_tools = [
            {
                "type": "function",
                "function": {
                    "name": "another_llm",
                    "description": "使用llm_tools可以调用其他的智能助手解决你的问题。请根据以下列表中的system_prompt选择你需要的智能助手："
                    + json.dumps(llm_tools_list, ensure_ascii=False, indent=4),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "智能助手的id"},
                            "type": {"type": "string", "description": "智能助手的类型，目前支持api和local两种类型。"},
                            "question": {"type": "string", "description": "问题描述，代表你想要解决的问题。"},
                        },
                        "required": ["id", "type", "question"],
                    },
                },
            }
        ]

        llm_tools_json = json.dumps(llm_tools, ensure_ascii=False, indent=4)
        if (user_prompt is None or user_prompt.strip() == "") and (images is None or images == []) and (user_history is None or user_history == [] or user_history.strip() == ""):
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            return (
                "",
                str(history),
                llm_tools_json,
                None,
                "",
            )
        else:
            try:
                if is_memory == "disable" or "clear party memory" in user_prompt:
                    with open(self.prompt_path, "w", encoding="utf-8") as f:
                        json.dump([{"role": "system", "content": system_prompt}], f, indent=4, ensure_ascii=False)
                    if "clear party memory" in user_prompt:
                        with open(self.prompt_path, "r", encoding="utf-8") as f:
                            history = json.load(f)
                        return (
                            "party memory has been cleared, please ask me again, powered by LLM party!",
                            str(history),
                            llm_tools_json,
                            None,
                            "",
                        )
                api_keys = load_api_keys(config_path)

                with open(self.prompt_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                if user_history != "" and user_history is not None:
                    try:
                        history = json.loads(user_history)
                    except:
                        pass
                history_temp = [history[0]]
                elements_to_keep = 2 * conversation_rounds
                if elements_to_keep < len(history) - 1:
                    history_temp += history[-elements_to_keep:]
                    history_copy = history[1:-elements_to_keep]
                else:
                    if len(history) > 1:
                        history_temp += history[1:]
                    history_copy = []
                if len(history_temp) > 1:
                    if history_temp[1]["role"] == "tool":
                        history_temp.insert(1, history[-elements_to_keep - 1])
                        if -elements_to_keep - 1 > 1:
                            history_copy = history[1 : -elements_to_keep - 1]
                        else:
                            history_copy = []
                history = history_temp
                for message in history:
                    if message["role"] == "system":
                        message["content"] = system_prompt
                if is_tools_in_sys_prompt == "enable":
                    tools_list = []
                    GPT_INSTRUCTION = ""
                    if tools is not None:
                        tools_dis = json.loads(tools)
                        tools = None
                        for tool_dis in tools_dis:
                            tools_list.append(tool_dis["function"])
                        tools_instructions = ""
                        tools_instruction_list = []
                        for tool in tools_list:
                            tools_instruction_list.append(tool["name"])
                            tools_instructions += (
                                str(tool["name"])
                                + ":"
                                + "Call this tool to interact with the "
                                + str(tool["name"])
                                + " API. What is the "
                                + str(tool["name"])
                                + " API useful for? "
                                + str(tool["description"])
                                + ". Parameters:"
                                + str(tool["parameters"])
                                + "Required parameters:"
                                + str(tool["parameters"]["required"])
                                + "\n"
                            )
                        REUTRN_FORMAT = '{"tool": "tool name", "parameters": {"parameter name": "parameter value"}}'
                        TOOL_EAXMPLE = 'You will receive a JSON string containing a list of callable tools. Please parse this JSON string and return a JSON object containing the tool name and tool parameters. Here is an example of the tool list:\n\n{"tools": [{"name": "plus_one", "description": "Add one to a number", "parameters": {"type": "object","properties": {"number": {"type": "string","description": "The number that needs to be changed, for example: 1","default": "1",}},"required": ["number"]}},{"name": "minus_one", "description": "Minus one to a number", "parameters": {"type": "object","properties": {"number": {"type": "string","description": "The number that needs to be changed, for example: 1","default": "1",}},"required": ["number"]}}]}\n\nBased on this tool list, generate a JSON object to call a tool. For example, if you need to add one to number 77, return:\n\n{"tool": "plus_one", "parameters": {"number": "77"}}\n\nPlease note that the above is just an example and does not mean that the plus_one and minus_one tools are currently available.'
                        GPT_INSTRUCTION = f"""
        Answer the following questions as best you can. You have access to the following APIs:
        {tools_instructions}

        Use the following format:
        ```tool_json
        {REUTRN_FORMAT}
        ```

        Please choose the appropriate tool according to the user's question. If you don't need to call it, please reply directly to the user's question. When the user communicates with you in a language other than English, you need to communicate with the user in the same language.

        When you have enough information from the tool results, respond directly to the user with a text message without having to call the tool again.
        """

                    for message in history:
                        if message["role"] == "system":
                            message["content"] = system_prompt
                            if tools_list != []:
                                message["content"] += "\n" + TOOL_EAXMPLE + "\n" + GPT_INSTRUCTION + "\n"

                if tools is not None:
                    print(tools)
                    tools = json.loads(tools)

                max_length = int(max_length)

                if file_content is not None:
                    for message in history:
                        if message["role"] == "system":
                            message["content"] += "\n以下是可以参考的已知信息:\n" + file_content
                if extra_parameters is not None and extra_parameters != {}:
                    response, history,reasoning_content = model.send(
                        user_prompt, temperature, max_length, history, tools, is_tools_in_sys_prompt,images,imgbb_api_key,img_URL,stream, **extra_parameters
                    )
                else:
                    response, history,reasoning_content = model.send(
                        user_prompt, temperature, max_length, history, tools, is_tools_in_sys_prompt,images,imgbb_api_key,img_URL,stream,
                    )
                # 修改prompt.json文件
                history_get = [history[0]]
                history_get.extend(history_copy)
                history_get.extend(history[1:])
                history = history_get
                with open(self.prompt_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=4, ensure_ascii=False)
                history = json.dumps(history, ensure_ascii=False,indent=4)
                global image_buffer
                if image_buffer != [] and image_buffer is not None:
                    image_out = image_buffer.clone()
                else:
                    image_out = None
                image_buffer = []
                return (
                    response,
                    history,
                    llm_tools_json,
                    image_out,
                    reasoning_content,
                )
            except Exception as ex:
                print(ex)
                return (
                    str(ex),
                    str(ex),
                    llm_tools_json,
                    None,
                    "",
                )

    @classmethod
    def original_IS_CHANGED(s):
        # 生成当前时间的哈希值
        hash_value = hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()
        return hash_value


def llm_chat(
    model, tokenizer, user_prompt, history, device, max_length, role="user", temperature=0.7, **extra_parameters
):
    history.append({"role": role, "content": user_prompt.strip()})
    # 检查是否已经设置了 chat_template
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        # 如果没有设置 chat_template，尝试使用 default_chat_template
        if hasattr(tokenizer, 'default_chat_template') and tokenizer.default_chat_template is not None:
            tokenizer.chat_template = tokenizer.default_chat_template
        else:
            print("chat_template is not set")
            model_inputs = tokenizer([user_prompt.strip()], return_tensors="pt").to(device)
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                eos_token_id=tokenizer.eos_token_id,
                **extra_parameters,  # Add the eos_token_id parameter
            )
            response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            history.append({"role": "assistant", "content": response})
            return response, history
    text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        **extra_parameters,  # Add the eos_token_id parameter
    )
    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    history.append({"role": "assistant", "content": response})
    return response, history

def llama_chat(
    model, processor, image, user_prompt, history, device, max_length, role="user", temperature=0.7, **extra_parameters
):
    if image !=[]:
        user_content = {
            "role": role,
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        }
        history.append(user_content)
        input_text = processor.apply_chat_template(history, add_generation_prompt=True)
        inputs = processor(image, input_text, return_tensors="pt").to(device)
    else:
        user_content = {
            "role": role,
            "content": [
                {"type": "text", "text": user_prompt},
            ],
        }
        history.append(user_content)
        input_text = processor.apply_chat_template(history, add_generation_prompt=True)
        inputs = processor(text=input_text, return_tensors="pt").to(device)
    

    output = model.generate(**inputs, max_new_tokens=max_length, temperature=temperature, **extra_parameters)
    response = processor.decode(output[0], skip_special_tokens=True)
    # 使用正则表达式提取最后一个助手的回答
    matches = re.findall(r'assistant\s*(.*?)(?=\s*(?:system|user|$))', response, re.DOTALL)
    if matches:
        assistant_output = matches[-1].strip()
    else:
        assistant_output = response.strip()
    history.append({"role": "assistant", "content":assistant_output})
    
    return assistant_output, history

def qwen_chat(
    model, processor, image, user_prompt, history, device, max_length, role="user", temperature=0.7, **extra_parameters
):
    if image !=[]:
        image_content = []
        for i in image:
            image_content.append(
                {
                    "type": "image",
                    "image": i,  
                }
            )
        image_content.append({"type": "text", "text": user_prompt})
        user_content = [
            {
                "role": "user",
                "content": image_content,
            }
        ]
        history.extend(user_content)
    else:
        user_content = [{
            "role": role,
            "content": [
                {"type": "text", "text": user_prompt},
            ],
        }]
        history.extend(user_content)
    # 准备推理输入
    text = processor.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(history)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    # 移除历史中的图片
    for i in range(len(history)-1, -1, -1):
        if history[i].get("role") == "user":
            for content in history[i]["content"]:
                if content.get("type") == "image":
                    history.pop(i)
                    history.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})
                    break

    # 生成输出
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_length, temperature=temperature, **extra_parameters)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
    response = output_text[0]
    # 使用正则表达式提取最后一个助手的回答
    matches = re.findall(r'assistant\s*(.*?)(?=\s*(?:system|user|$))', response, re.DOTALL)
    if matches:
        assistant_output = matches[-1].strip()
    else:
        assistant_output = response.strip()
    history.append({"role": "assistant", "content":assistant_output})
    
    return assistant_output, history

def convert_pil_images_to_base64(pil_images):
    base64_images = []
    for pil_img in pil_images:
        # 确保图像格式为PNG或其他你想要转换的格式
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)  # 移动到字节流的开始
        
        # 将字节流转换为Base64字符串
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # 构造data URL scheme格式（可选）
        img_base64 = f"data:image/png;base64,{img_str}"
        base64_images.append(img_base64)
    
    return base64_images

def ds_chat(
    model, processor, image, user_prompt, history, device, max_length, role="<|User|>", temperature=0.7, **extra_parameters
):
    tokenizer = processor.tokenizer
    if image !=[]:
        base64_images = convert_pil_images_to_base64(image)
        user_content = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{user_prompt}",
                "images": base64_images,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
    else:
        user_content = [
            {
                "role": "<|User|>",
                "content": f"{user_prompt}",
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
    try :
        from janus.utils.io import load_pil_images
        # specify the path to the model
        # load images and prepare for inputs
        pil_images = load_pil_images(user_content)
        prepare_inputs = processor(
            conversations=user_content, images=pil_images, force_batchify=True
        ).to(device)

        # # run image encoder to get the image embeddings
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        # # run the model to get the response
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_length, 
            temperature=temperature,
            do_sample=False,
            use_cache=True,
            **extra_parameters,
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        clean_pattern = r'<\|.*?\|>'
        clean_answer = re.sub(clean_pattern, '', answer).strip()
        history.append({"role": "assistant", "content":clean_answer})
    except Exception as e:
        print(e)
        clean_answer = str(e)
    
    return clean_answer, history

class LLM_local_loader:
    def __init__(self):
        self.id = hash(str(self))
        self.device = ""
        self.dtype = ""
        self.model_name_or_path = ""
        self.model = ""
        self.tokenizer = ""
        self.is_locked = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name_or_path": ("STRING", {"default": "","tooltip": "You can provide the absolute path to the model folder, or you can enter the repo ID from Hugging Face, for example: lllyasviel/omost-llama-3-8b-4bits."}),
                "device": (
                    ["auto", "cuda", "cpu", "mps"],
                    {
                        "default": "auto",
                        "tooltip": "The device to use for the model. If 'auto', it will use 'cuda' if available, otherwise 'mps' if available, otherwise 'cpu'.",
                    },
                ),
                "dtype": (
                    ["auto","float32", "float16","bfloat16", "int8", "int4"],
                    {
                        "default": "auto",
                        "tooltip": "The data type to use for the model. If 'float32', it will use 'float32', otherwise 'float16', 'bfloat16', 'int8', 'int4'.",
                    },
                ),
                "is_locked": ("BOOLEAN", {"default": True, "tooltip": "Whether the model is locked or not.When enabled, it prevents the model from being loaded multiple times. When disabled, it can be used in conjunction with clearing the GPU memory node to reload the model."}),
            }
        }

    RETURN_TYPES = (
        "CUSTOM",
        "CUSTOM",
    )
    RETURN_NAMES = (
        "model",
        "tokenizer",
    )
    OUTPUT_TOOLTIPS = (
        "The loaded model.",
        "The loaded tokenizer.",
    )
    FUNCTION = "chatbot"
    DESCRIPTION = "Load a local model from a given path or Hugging Face repo ID.The model must not be in GGUF format but instead be an LLM model stored in a folder."

    # OUTPUT_NODE = False

    CATEGORY = "大模型派对（llm_party）/模型加载器（model loader）"

    def chatbot(self, model_name_or_path, device, dtype, is_locked=True):
        self.is_locked = is_locked
        if self.is_locked == False:
            setattr(LLM_local_loader, "IS_CHANGED", LLM_local_loader.original_IS_CHANGED)
        else:
            # 如果方法存在，则删除
            if hasattr(LLM_local_loader, "IS_CHANGED"):
                delattr(LLM_local_loader, "IS_CHANGED")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        if (
            self.device != device
            or self.dtype != dtype
            or self.model_name_or_path != model_name_or_path
            or is_locked == False
        ):
            del self.model
            del self.tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            # 对于 CPU 和 MPS 设备，不需要清空 CUDA 缓存
            elif self.device == "cpu" or self.device == "mps":
                gc.collect()
            self.model = ""
            self.tokenizer = ""
            self.model_name_or_path = model_name_or_path
            self.device = device
            self.dtype = dtype
            model_kwargs = {
                'device_map': device,
            }

            if dtype == "float16":
                model_kwargs['torch_dtype'] = torch.float16
            elif dtype == "bfloat16":
                model_kwargs['torch_dtype'] = torch.bfloat16
            elif dtype in ["int8", "int4"]:
                model_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=(dtype == "int8"), load_in_4bit=(dtype == "int4"))

            config = AutoConfig.from_pretrained(model_name_or_path, **model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

            if config.model_type == "t5":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **model_kwargs)
            elif config.model_type in ["gpt2", "gpt_refact", "gemma", "llama", "mistral", "qwen2", "chatglm"]:
                self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            self.model = self.model.eval()
        return (
            self.model,
            self.tokenizer,
        )

    @classmethod
    def original_IS_CHANGED(s):
        # 生成当前时间的哈希值
        hash_value = hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()
        return hash_value

LLM_dir=os.path.join(current_dir_path, "model","LLM")
# 获取LLM_dir文件夹中的所有文件夹
LLM_list = [f for f in os.listdir(LLM_dir) if os.path.isdir(os.path.join(LLM_dir, f))]
class easy_LLM_local_loader:
    def __init__(self):
        self.id = hash(str(self))
        self.device = ""
        self.dtype = ""
        self.model_name_or_path = ""
        self.model = ""
        self.tokenizer = ""
        self.is_locked = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name_or_path": (LLM_list, {"default": "","tooltip": "Select your model files from custom_nodes\comfyui_LLM_party\model\LLM."}),
                "device": (
                    ["auto", "cuda", "cpu", "mps"],
                    {
                        "default": "auto",
                        "tooltip": "Select the device to load the model on. 'auto' will use the best available device.",
                    },
                ),
                "dtype": (
                    ["auto","float32", "float16","bfloat16", "int8", "int4"],
                    {
                        "default": "auto",
                        "tooltip": "The data type to use for the model. If 'float32', it will use 'float32', otherwise 'float16', 'bfloat16', 'int8', 'int4'.",
                    },
                ),
                "is_locked": ("BOOLEAN", {"default": True, "tooltip": "Whether the model is locked or not.When enabled, it prevents the model from being loaded multiple times. When disabled, it can be used in conjunction with clearing the GPU memory node to reload the model."}),
            }
        }

    RETURN_TYPES = (
        "CUSTOM",
        "CUSTOM",
    )
    RETURN_NAMES = (
        "model",
        "tokenizer",
    )
    OUTPUT_TOOLTIPS = (
        "The loaded model.",
        "The loaded tokenizer."
    )
    DESCRIPTION = "Load a local model from custom_nodes\comfyui_LLM_party\model\LLM."

    FUNCTION = "chatbot"

    # OUTPUT_NODE = False

    CATEGORY = "大模型派对（llm_party）/模型加载器（model loader）"

    def chatbot(self, model_name_or_path, device, dtype, is_locked=True):
        model_name_or_path=os.path.join(LLM_dir,model_name_or_path)
        self.is_locked = is_locked
        if self.is_locked == False:
            setattr(easy_LLM_local_loader, "IS_CHANGED", easy_LLM_local_loader.original_IS_CHANGED)
        else:
            # 如果方法存在，则删除
            if hasattr(easy_LLM_local_loader, "IS_CHANGED"):
                delattr(easy_LLM_local_loader, "IS_CHANGED")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        if (
            self.device != device
            or self.dtype != dtype
            or self.model_name_or_path != model_name_or_path
            or is_locked == False
        ):
            del self.model
            del self.tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            # 对于 CPU 和 MPS 设备，不需要清空 CUDA 缓存
            elif self.device == "cpu" or self.device == "mps":
                gc.collect()
            self.model = ""
            self.tokenizer = ""
            self.model_name_or_path = model_name_or_path
            self.device = device
            self.dtype = dtype
            model_kwargs = {
                'device_map': device,
            }

            if dtype == "float16":
                model_kwargs['torch_dtype'] = torch.float16
            elif dtype == "bfloat16":
                model_kwargs['torch_dtype'] = torch.bfloat16
            elif dtype in ["int8", "int4"]:
                model_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=(dtype == "int8"), load_in_4bit=(dtype == "int4"))

            config = AutoConfig.from_pretrained(model_name_or_path, **model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

            if config.model_type == "t5":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **model_kwargs)
            elif config.model_type in ["gpt2", "gpt_refact", "gemma", "llama", "mistral", "qwen2", "chatglm"]:
                self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            self.model = self.model.eval()
        return (
            self.model,
            self.tokenizer,
        )

    @classmethod
    def original_IS_CHANGED(s):
        # 生成当前时间的哈希值
        hash_value = hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()
        return hash_value


class LLM_local:
    def __init__(self):
        # 生成一个hash值作为id
        current_time = datetime.datetime.now()
        self.id = current_time.strftime("%Y_%m_%d_%H_%M_%S") + str(hash(random.randint(0, 1000000)))
        global instances
        instances.append(self)
        # 构建prompt.json的绝对路径，如果temp文件夹不存在就创建
        current_dir_path = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(current_dir_path, "temp"), exist_ok=True)
        self.prompt_path = os.path.join(current_dir_path, "temp", str(self.id) + ".json")
        # 如果文件不存在，创建prompt.json文件，存在就覆盖文件
        if not os.path.exists(self.prompt_path):
            with open(self.prompt_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{"role": "system", "content": "你是一个强大的人工智能助手。"}], f, indent=4, ensure_ascii=False
                )
        self.tool_data = {"id": self.id, "system_prompt": "", "type": "local"}
        self.list = []
        self.added_to_list = False
        self.is_locked = "disable"
        self.images = []

    @classmethod
    def INPUT_TYPES(s):
        temp_path = os.path.join(current_dir_path, "temp")
        full_paths = [os.path.join(temp_path, f) for f in os.listdir(temp_path)]
        full_paths.sort(key=os.path.getmtime, reverse=True)
        paths = [os.path.basename(f) for f in full_paths]
        paths.insert(0, "")
        return {
            "required": {
                "model": ("CUSTOM", {"tooltip": "The model to use for the LLM."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "你一个强大的人工智能助手。","tooltip": "System prompt, used to describe the behavior of the model and the expected output format."}),
                "user_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "你好",
                        "tooltip": "User prompt, used to describe the user's request and the expected output format.",
                    },
                ),
                "model_type": (
                    ["LLM","LLM-GGUF", "VLM-GGUF", "VLM(llama-v)", "VLM(qwen-vl)","VLM(deepseek-janus-pro)"],
                    {
                        "default": "LLM",
                        "tooltip": "The type of model to use for the LLM. LLM: Language Model, VLM: Vision Language Model, GGUF: Generalized GPT-4 Unified Framework",
                    },
                ),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1,"tooltip": "The temperature parameter controls the randomness of the model's output. A higher temperature will result in more random and diverse responses, while a lower temperature will result in more focused and deterministic responses."}),
                "max_length":("INT", {"default": 512, "min": 256, "max": 128000, "step": 128,"tooltip": "The maximum length of the output text."}),
                "is_memory": (["enable", "disable"], {"default": "enable","tooltip": "Whether to enable memory for the LLM."}),
                "is_locked": (["enable", "disable"], {"default": "disable", "tooltip": "Whether to directly output the result from the last output."}),
                "main_brain": (["enable", "disable"], {"default": "enable", "tooltip": "If this option is disabled, the LLM will become a tool that can be invoked by other LLMs."}),
            },
            "optional": {
                "tokenizer": ("CUSTOM", {"tooltip":"The tokenizer to use for the LLM."}),
                "image": ("IMAGE", {"forceInput": True, "tooltip": "Upload images here."}),
                "system_prompt_input": ("STRING", {"forceInput": True, "tooltip": "System prompt input, used to describe the system's request and the expected output format."}),
                "user_prompt_input": ("STRING", {"forceInput": True, "tooltip": "User prompt input, used to describe the user's request and the expected output format."}),
                "tools": ("STRING", {"forceInput": True,"tooltip": "Tool list, used to describe the tools that the model can invoke."}),
                "file_content": ("STRING", {"forceInput": True, "tooltip": "Input the contents of the file here."}),
                "conversation_rounds": ("INT", {"default": 100, "min": 1, "max": 10000, "step": 1, "tooltip": "The maximum number of dialogue turns that the LLM can see in the history records, where one question and one answer constitute one turn."}),
                "historical_record": (paths, {"default": "", "tooltip": "The dialogue history file is optional; if not selected and left empty, a new dialogue history file will be automatically created."}),
                "is_enable": ("BOOLEAN", {"default": True, "tooltip": "Whether to enable the LLM."}),
                "extra_parameters": ("DICT", {"forceInput": True, "tooltip": "Extra parameters for the LLM."}),
                "user_history": ("STRING", {"forceInput": True, "tooltip": "User history, you can directly input a JSON string containing multiple rounds of dialogue here."}),
                "is_enable_system_role": (["enable", "disable"], {"default": "enable","tooltip": "Whether to enable the system role for the LLM."}),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "IMAGE",
    )
    RETURN_NAMES = (
        "assistant_response",
        "history",
        "tool",
        "image",
    )
    OUTPUT_TOOLTIPS = (
        "The assistant's response to the user's request.",
        "The dialogue history",
        "This interface will connect this LLM as a tool to other LLMs.",
        "Images generated or fetched by the LLM.",
    )
    DESCRIPTION = "The local version of the model chain, compatible with all local interfaces."

    FUNCTION = "chatbot"

    # OUTPUT_NODE = False

    CATEGORY = "大模型派对（llm_party）/模型链（model_chain）"

    def chatbot(
        self,
        user_prompt,
        main_brain,
        system_prompt,
        model_type,
        model,
        tokenizer=None,
        temperature=0.7,
        max_length=2048,
        is_memory="enable",
        is_locked="disable",
        system_prompt_input="",
        user_prompt_input="",
        tools=None,
        file_content=None,
        image=None,
        conversation_rounds=100,
        historical_record=None,
        is_enable=True,
        extra_parameters=None,
        user_history=None,
        is_enable_system_role="enable",
    ):
        if not is_enable:
            return (
                None,
                None,
                None,
                None,
            )
        self.list = [
            main_brain,
            system_prompt,
            model_type,
            model,
            tokenizer,
            temperature,
            max_length,
            is_memory,
            is_locked,
            system_prompt_input,
            user_prompt_input,
            tools,
            file_content,
            image,
            conversation_rounds,
            historical_record,
            is_enable,
            extra_parameters,
            user_history,
            is_enable_system_role,
        ]
        if user_prompt is None:
            user_prompt = user_prompt_input
        elif user_prompt_input is not None:
            user_prompt = user_prompt + user_prompt_input
        if historical_record != "":
            temp_path = os.path.join(current_dir_path, "temp")
            self.prompt_path = os.path.join(temp_path, historical_record)
        self.tool_data["system_prompt"] = system_prompt
        if system_prompt_input is not None and system_prompt is not None:
            system_prompt = system_prompt + system_prompt_input
        elif system_prompt is None:
            system_prompt = system_prompt_input
        global llm_tools_list, llm_tools
        if main_brain == "disable":
            if not self.added_to_list:
                llm_tools_list.append(self.tool_data)
                self.added_to_list = True
        self.is_locked = is_locked
        if self.is_locked == "disable":
            setattr(LLM_local, "IS_CHANGED", LLM_local.original_IS_CHANGED)
        else:
            # 如果方法存在，则删除
            if hasattr(LLM_local, "IS_CHANGED"):
                delattr(LLM_local, "IS_CHANGED")
        llm_tools = [
            {
                "type": "function",
                "function": {
                    "name": "another_llm",
                    "description": "使用llm_tools可以调用其他的智能助手解决你的问题。请根据以下列表中的system_prompt选择你需要的智能助手："
                    + json.dumps(llm_tools_list, ensure_ascii=False, indent=4),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "智能助手的id"},
                            "type": {"type": "string", "description": "智能助手的类型，目前支持api和local两种类型。"},
                            "question": {"type": "string", "description": "问题描述，代表你想要解决的问题。"},
                        },
                        "required": ["id", "type", "question"],
                    },
                },
            }
        ]
        llm_tools_json = json.dumps(llm_tools, ensure_ascii=False, indent=4)
        if (user_prompt is None or user_prompt.strip() == "") and (image is None or image == []) and (user_history is None or user_history == [] or user_history.strip() == ""):
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            return (
                "",
                str(history),
                llm_tools_json,
                None,
            )
        else:
            try:
                if is_memory == "disable" or "clear party memory" in user_prompt:
                    self.images= []
                    with open(self.prompt_path, "w", encoding="utf-8") as f:
                        json.dump([{"role": "system", "content": system_prompt}], f, indent=4, ensure_ascii=False)
                    if "clear party memory" in user_prompt:
                        with open(self.prompt_path, "r", encoding="utf-8") as f:
                            history = json.load(f)
                        return (
                            "party memory has been cleared, please ask me again, powered by LLM party!",
                            str(history),
                            llm_tools_json,
                            None,
                        )
                with open(self.prompt_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                if user_history != "" and user_history is not None:
                    try:
                        history = json.loads(user_history)
                    except:
                        pass
                history_temp = [history[0]]
                elements_to_keep = 2 * conversation_rounds
                if elements_to_keep < len(history) - 1:
                    history_temp += history[-elements_to_keep:]
                    history_copy = history[1:-elements_to_keep]
                else:
                    if len(history) > 1:
                        history_temp += history[1:]
                    history_copy = []
                if len(history_temp) > 1:
                    if history_temp[1]["role"] == "tool":
                        history_temp.insert(1, history[-elements_to_keep - 1])
                        if -elements_to_keep - 1 > 1:
                            history_copy = history[1 : -elements_to_keep - 1]
                        else:
                            history_copy = []
                history = history_temp
                tools_list = []
                GPT_INSTRUCTION = ""
                if tools is not None:
                    tools_dis = json.loads(tools)
                    for tool_dis in tools_dis:
                        tools_list.append(tool_dis["function"])
                    tools_instructions = ""
                    tools_instruction_list = []
                    for tool in tools_list:
                        tools_instruction_list.append(tool["name"])
                        tools_instructions += (
                            str(tool["name"])
                            + ":"
                            + "Call this tool to interact with the "
                            + str(tool["name"])
                            + " API. What is the "
                            + str(tool["name"])
                            + " API useful for? "
                            + str(tool["description"])
                            + ". Parameters:"
                            + str(tool["parameters"])
                            + "Required parameters:"
                            + str(tool["parameters"]["required"])
                            + "\n"
                        )
                    REUTRN_FORMAT = '{"tool": "tool name", "parameters": {"parameter name": "parameter value"}}'
                    TOOL_EAXMPLE = 'You will receive a JSON string containing a list of callable tools. Please parse this JSON string and return a JSON object containing the tool name and tool parameters. Here is an example of the tool list:\n\n{"tools": [{"name": "plus_one", "description": "Add one to a number", "parameters": {"type": "object","properties": {"number": {"type": "string","description": "The number that needs to be changed, for example: 1","default": "1",}},"required": ["number"]}},{"name": "minus_one", "description": "Minus one to a number", "parameters": {"type": "object","properties": {"number": {"type": "string","description": "The number that needs to be changed, for example: 1","default": "1",}},"required": ["number"]}}]}\n\nBased on this tool list, generate a JSON object to call a tool. For example, if you need to add one to number 77, return:\n\n{"tool": "plus_one", "parameters": {"number": "77"}}\n\nPlease note that the above is just an example and does not mean that the plus_one and minus_one tools are currently available.'
                    GPT_INSTRUCTION = f"""
    Answer the following questions as best you can. You have access to the following APIs:
    {tools_instructions}

    Use the following format:
    ```tool_json
    {REUTRN_FORMAT}
    ```

    Please choose the appropriate tool according to the user's question. If you don't need to call it, please reply directly to the user's question. When the user communicates with you in a language other than English, you need to communicate with the user in the same language.

    When you have enough information from the tool results, respond directly to the user with a text message without having to call the tool again.
    """

                for message in history:
                    if message["role"] == "system":
                        message["content"] = system_prompt
                        if tools_list != []:
                            if model_type in ["LLM", "VLM(testing)"]:
                                message["content"] += "\n" + TOOL_EAXMPLE + "\n" + GPT_INSTRUCTION + "\n"
                if tools is not None:
                    print(tools)
                    tools = json.loads(tools)

                def is_valid_json(json_str):
                    try:
                        json.loads(json_str)
                        return True
                    except json.JSONDecodeError:
                        return False

                max_length = int(max_length)
                if file_content is not None:
                    for message in history:
                        if message["role"] == "system":
                            message["content"] += "\n以下是可以参考的已知信息:\n" + file_content
                if is_enable_system_role == "disable":
                    if history[0]["role"] == "system":
                        history[0]["role"]= "user"
                # 获得model存放的设备
                if model_type not in ["VLM-GGUF", "LLM-GGUF"]:
                    device = next(model.parameters()).device
                if model_type == "LLM":
                    if extra_parameters is not None and extra_parameters != {}:
                        response, history = llm_chat(
                            model,
                            tokenizer,
                            user_prompt,
                            history,
                            device,
                            max_length,
                            temperature=temperature,
                            **extra_parameters,
                        )
                    else:
                        response, history = llm_chat(
                            model, tokenizer, user_prompt, history, device, max_length, temperature=temperature
                        )
                    # 正则表达式匹配
                    pattern = r'\{\s*"tool":\s*"(.*?)",\s*"parameters":\s*\{(.*?)\}\s*\}'
                    while re.search(pattern, response, re.DOTALL) != None:
                        match = re.search(pattern, response, re.DOTALL)
                        tool = match.group(1)
                        parameters = match.group(2)
                        json_str = '{"tool": "' + tool + '", "parameters": {' + parameters + "}}"
                        history.append({"role": "function_call", "content": json_str})
                        print("正在调用" + tool + "工具")
                        parameters = json.loads("{" + parameters + "}")
                        results = dispatch_tool(tool, parameters)
                        print(results)
                        if extra_parameters is not None and extra_parameters != {}:
                            response, history = llm_chat(
                                model,
                                tokenizer,
                                results,
                                history,
                                device,
                                max_length,
                                role="observation",
                                temperature=temperature,
                                **extra_parameters,
                            )
                        else:
                            response, history = llm_chat(
                                model,
                                tokenizer,
                                results,
                                history,
                                device,
                                max_length,
                                role="observation",
                                temperature=temperature,
                            )
                elif model_type == "VLM-GGUF":
                    if image is not None:
                        user_content = {
                            "role": "user",
                            "content": []
                        }
                        for img_VLMG in image:
                            pil_image = ToPILImage()(img_VLMG.permute(2, 0, 1))
                            buffer = io.BytesIO()
                            pil_image.save(buffer, format="PNG")
                            base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                            user_content["content"].append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_string}"}
                            })
                        user_content["content"].append({"type": "text", "text": user_prompt})

                        history.append(user_content)
                        if extra_parameters is not None and extra_parameters != {}:
                            response = model.create_chat_completion(
                                messages=history,
                                temperature=temperature,
                                max_tokens=max_length,
                                **extra_parameters,
                            )
                        else:
                            response = model.create_chat_completion(
                                messages=history,
                                temperature=temperature,
                                max_tokens=max_length,
                            )
                        response = f"{response['choices'][0]['message']['content']}"
                        print(response)
                        assistant_content = {"role": "assistant", "content": response}
                        history.append(assistant_content)
                    else:
                        user_content = {"role": "user", "content": user_prompt}
                        history.append(user_content)
                        if extra_parameters is not None and extra_parameters != {}:
                            response = model.create_chat_completion(
                                messages=history,
                                temperature=temperature,
                                max_tokens=max_length,
                                **extra_parameters,
                            )
                        else:
                            response = model.create_chat_completion(
                                messages=history,
                                temperature=temperature,
                                max_tokens=max_length,
                            )
                        response = f"{response['choices'][0]['message']['content']}"
                        assistant_content = {"role": "assistant", "content": response}
                        history.append(assistant_content)
                elif model_type == "LLM-GGUF":
                    user_content = {"role": "user", "content": user_prompt}
                    history.append(user_content)
                    if extra_parameters is not None and extra_parameters != {}:
                        response = model.create_chat_completion(
                            messages=history,
                            temperature=temperature,
                            max_tokens=max_length,
                            tools=tools,
                            **extra_parameters,
                        )
                    else:
                        response = model.create_chat_completion(
                            messages=history,
                            temperature=temperature,
                            max_tokens=max_length,
                            tools=tools,
                        )
                    assistant_message = f"{response['choices'][0]['message']['content']}"
                    while assistant_message.startswith('{"name":'):
                        response_content = json.loads(assistant_message)
                        print("正在调用" + response_content['name'] + "工具")
                        results = dispatch_tool(response_content['name'], response_content['parameters'])
                        print(results)
                        assistant_content = {"role": "function_call", "content": assistant_message}
                        history.append(assistant_content)    
                        history.append(
                            {
                                "role": "observation",
                                "content": results,
                            }
                        )
                        if extra_parameters is not None and extra_parameters != {}:
                            response = model.create_chat_completion(
                                messages=history,
                                temperature=temperature,
                                max_tokens=max_length,
                                tools=tools,
                                **extra_parameters,
                            )
                        else:
                            response = model.create_chat_completion(
                                messages=history,
                                temperature=temperature,
                                max_tokens=max_length,
                                tools=tools,
                            )
                        print(response)   
                        assistant_message = f"{response['choices'][0]['message']['content']}"
                    assistant_content = {"role": "assistant", "content": assistant_message}
                    response= assistant_message
                    history.append(assistant_content)    
                elif model_type =="VLM(llama-v)":
                    if image is not None:
                        for img_VLM in image:
                            pil_image = ToPILImage()(img_VLM.permute(2, 0, 1))
                            self.images.append(pil_image)
                    if extra_parameters is not None and extra_parameters != {}:
                        response, history = llama_chat(
                            model,
                            tokenizer,
                            self.images,
                            user_prompt,
                            history,
                            device,
                            max_length,
                            temperature=temperature,
                            **extra_parameters,
                        )
                    else:
                        response, history = llama_chat(
                            model, tokenizer, self.images, user_prompt, history, device, max_length, temperature=temperature
                        )
                    # 正则表达式匹配
                    pattern = r'\{\s*"tool":\s*"(.*?)",\s*"parameters":\s*\{(.*?)\}\s*\}'
                    while re.search(pattern, response, re.DOTALL) != None:
                        match = re.search(pattern, response, re.DOTALL)
                        tool = match.group(1)
                        parameters = match.group(2)
                        json_str = '{"tool": "' + tool + '", "parameters": {' + parameters + "}}"
                        history.append({"role": "function_call", "content": json_str})
                        print("正在调用" + tool + "工具")
                        parameters = json.loads("{" + parameters + "}")
                        results = dispatch_tool(tool, parameters)
                        print(results)
                        if extra_parameters is not None and extra_parameters != {}:
                            response, history = llama_chat(
                                model,
                                tokenizer,
                                self.images, # image,
                                results,
                                history,
                                device,
                                max_length,
                                role="observation",
                                temperature=temperature,
                                **extra_parameters,
                            )
                        else:
                            response, history = llama_chat(
                                model,
                                tokenizer,
                                self.images, # image,
                                results,
                                history,
                                device,
                                max_length,
                                role="observation",
                                temperature=temperature,
                            )
                elif model_type =="VLM(qwen-vl)":     
                    if image is not None:
                        for img_VLM in image:
                            pil_image = ToPILImage()(img_VLM.permute(2, 0, 1))
                            self.images.append(pil_image)
                    if extra_parameters is not None and extra_parameters != {}:
                        response, history = qwen_chat(
                            model,
                            tokenizer,
                            self.images,
                            user_prompt,
                            history,
                            device,
                            max_length,
                            temperature=temperature,
                            **extra_parameters,
                        )
                    else:
                        response, history = qwen_chat(
                            model, tokenizer, self.images, user_prompt, history, device, max_length, temperature=temperature
                        )
                    # 正则表达式匹配
                    pattern = r'\{\s*"tool":\s*"(.*?)",\s*"parameters":\s*\{(.*?)\}\s*\}'
                    while re.search(pattern, response, re.DOTALL) != None:
                        match = re.search(pattern, response, re.DOTALL)
                        tool = match.group(1)
                        parameters = match.group(2)
                        json_str = '{"tool": "' + tool + '", "parameters": {' + parameters + "}}"
                        history.append({"role": "function_call", "content": json_str})
                        print("正在调用" + tool + "工具")
                        parameters = json.loads("{" + parameters + "}")
                        results = dispatch_tool(tool, parameters)
                        print(results)
                        if extra_parameters is not None and extra_parameters != {}:
                            response, history = qwen_chat(
                                model,
                                tokenizer,
                                self.images, # image,
                                results,
                                history,
                                device,
                                max_length,
                                role="observation",
                                temperature=temperature,
                                **extra_parameters,
                            )
                        else:
                            response, history = qwen_chat(
                                model,
                                tokenizer,
                                self.images, # image,
                                results,
                                history,
                                device,
                                max_length,
                                role="observation",
                                temperature=temperature,
                            )    
                elif model_type =="VLM(deepseek-janus-pro)":
                    if image is not None:
                        for img_VLM in image:
                            pil_image = ToPILImage()(img_VLM.permute(2, 0, 1))
                            self.images.append(pil_image)
                    if extra_parameters is not None and extra_parameters != {}:
                        response, history = ds_chat(
                            model,
                            tokenizer,
                            self.images,
                            user_prompt,
                            history,
                            device,
                            max_length,
                            temperature=temperature,
                            **extra_parameters,
                        )
                    else:
                        response, history = ds_chat(
                            model, tokenizer, self.images, user_prompt, history, device, max_length, temperature=temperature
                        )
                    # 正则表达式匹配
                    pattern = r'\{\s*"tool":\s*"(.*?)",\s*"parameters":\s*\{(.*?)\}\s*\}'
                    while re.search(pattern, response, re.DOTALL) != None:
                        match = re.search(pattern, response, re.DOTALL)
                        tool = match.group(1)
                        parameters = match.group(2)
                        json_str = '{"tool": "' + tool + '", "parameters": {' + parameters + "}}"
                        history.append({"role": "function_call", "content": json_str})
                        print("正在调用" + tool + "工具")
                        parameters = json.loads("{" + parameters + "}")
                        results = dispatch_tool(tool, parameters)
                        print(results)
                        if extra_parameters is not None and extra_parameters != {}:
                            response, history = ds_chat(
                                model,
                                tokenizer,
                                self.images, # image,
                                results,
                                history,
                                device,
                                max_length,
                                role="observation",
                                temperature=temperature,
                                **extra_parameters,
                            )
                        else:
                            response, history = ds_chat(
                                model,
                                tokenizer,
                                self.images, # image,
                                results,
                                history,
                                device,
                                max_length,
                                role="observation",
                                temperature=temperature,
                            )   
                    self.images = []
                print(response)
                # 修改prompt.json文件
                history_get = [history[0]]
                history_get.extend(history_copy)
                history_get.extend(history[1:])
                history = history_get
                if is_enable_system_role == "disable":
                    if history[0]["role"] == "user":
                        history[0]["role"]= "system"
                with open(self.prompt_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=4, ensure_ascii=False)
                for his in history:
                    if his["role"] == "user":
                        # 如果his["content"]是个列表，则只保留"type" : "text"时的"text"属性内容
                        if isinstance(his["content"], list):
                            for item in his["content"]:
                                if item.get("type") == "text" and item.get("text"):
                                    his["content"] = item["text"]
                                    break
                historys = ""
                # 将history中的消息转换成便于用户阅读的markdown格式
                for his in history:
                    if his["role"] == "user":
                        historys += f"**User:** {his['content']}\n\n"
                    elif his["role"] == "assistant":
                        historys += f"**Assistant:** {his['content']}\n\n"
                    elif his["role"] == "system":
                        historys += f"**System:** {his['content']}\n\n"
                    elif his["role"] == "observation":
                        historys += f"**Observation:** {his['content']}\n\n"
                    elif his["role"] == "tool":
                        historys += f"**Tool:** {his['content']}\n\n"
                    elif his["role"] == "function":
                        historys += f"**Function:** {his['content']}\n\n"

                history = str(historys)
                global image_buffer
                if image_buffer != []:
                    image_out = image_buffer.clone()
                else:
                    image_out = None
                image_buffer = []
                return (
                    response,
                    history,
                    llm_tools_json,
                    image_out,
                )
            except Exception as ex:
                print(ex)
                return (
                    str(ex),
                    str(ex),
                    llm_tools_json,
                    None,
                )

    @classmethod
    def original_IS_CHANGED(s):
        # 生成当前时间的哈希值
        hash_value = hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()
        return hash_value




NODE_CLASS_MAPPINGS = {
    "LLM": LLM,
    "LLM_local": LLM_local,
    "LLM_api_loader": LLM_api_loader,
    # "genai_api_loader":genai_api_loader,
    "LLM_local_loader": LLM_local_loader,
    "easy_LLM_local_loader": easy_LLM_local_loader,
    "easy_LLM_api_loader":easy_LLM_api_loader,
    "load_ebd":load_ebd,
    "load_file": load_file,
    "load_persona": load_persona,
    "embeddings_function": embeddings_function,
    "classify_persona": classify_persona,
    "classify_function": classify_function,
    "classify_persona_plus": classify_persona_plus,
    "classify_function_plus": classify_function_plus,
    "tool_combine": tool_combine,
    "tool_combine_plus": tool_combine_plus,
    "time_tool": time_tool,
    "accuweather_tool": accuweather_tool,
    "google_tool": google_tool,
    "bing_tool": bing_tool,
    "check_web_tool": check_web_tool,
    "file_combine": file_combine,
    "file_combine_plus": file_combine_plus,
    "string_combine": string_combine,
    "string_combine_plus": string_combine_plus,
    "start_dialog": start_dialog,
    "end_dialog": end_dialog,
    # "interpreter_tool": interpreter_tool,
    "ebd_tool": ebd_tool,
    "custom_persona": custom_persona,
    "start_workflow": start_workflow,
    "end_workflow": end_workflow,
    "CLIPTextEncode_party": CLIPTextEncode_party,
    "KSampler_party": KSampler_party,
    "VAEDecode_party": VAEDecode_party,
    "string_logic": string_logic,
    "show_text_party": show_text_party,
    "load_url": load_url,
    "load_file_folder": load_file_folder,
    "api_tool": api_tool,
    "wikipedia_tool": wikipedia_tool,
    "load_wikipedia": load_wikipedia,
    "workflow_transfer": workflow_transfer,
    "About_us": About_us,
    "workflow_tool": workflow_tool,
    "github_tool": github_tool,
    "work_wechat_tool": work_wechat_tool,
    "work_wechat": work_wechat,
    "Dingding_tool": Dingding_tool,
    "Dingding": Dingding,
    "feishu_tool": feishu_tool,
    "feishu": feishu,
    "substring": substring,
    "openai_tts": openai_tts,
    "load_name": load_name,
    "omost_decode": omost_decode,
    "omost_setting": omost_setting,
    "keyword_tool": keyword_tool,
    "load_keyword": load_keyword,
    "story_json_tool": story_json_tool,
    "KG_json_toolkit_developer": KG_json_toolkit_developer,
    "KG_json_toolkit_user": KG_json_toolkit_user,
    "KG_csv_toolkit_developer": KG_csv_toolkit_developer,
    "KG_csv_toolkit_user": KG_csv_toolkit_user,
    "replace_string": replace_string,
    "KG_neo_toolkit_developer": KG_neo_toolkit_developer,
    "KG_neo_toolkit_user": KG_neo_toolkit_user,
    "translate_persona": translate_persona,
    "load_excel": load_excel,
    "text_iterator": text_iterator,
    "image_iterator": image_iterator,
    "google_loader": google_loader,
    "bing_loader": bing_loader,
    "api_function": api_function,
    "parameter_function": parameter_function,
    "get_string": get_string,
    "parameter_combine": parameter_combine,
    "parameter_combine_plus": parameter_combine_plus,
    "list_append": list_append,
    "list_append_plus": list_append_plus,
    "list_extend": list_extend,
    "list_extend_plus": list_extend_plus,
    "clear_model": clear_model,
    "save_ebd_database": save_ebd_database,
    "json2text": json2text,
    # "interpreter_function": interpreter_function,
    "load_img_path": load_img_path,
    "img2path": img2path,
    "load_int": load_int,
    "none2false": none2false,
    "bool_logic": bool_logic,
    "duckduckgo_tool":duckduckgo_tool,
    "duckduckgo_loader":duckduckgo_loader,
    "flux_persona":flux_persona,
    "clear_file":clear_file,
    "workflow_transfer_v2":workflow_transfer_v2,
    "str2float":str2float,
    "json_iterator":json_iterator,
    "Lorebook":Lorebook,
    "text_writing":text_writing,
    "str2int":str2int,
    "any2str":any2str,
    "start_anything":start_anything,
    "end_anything":end_anything,
    "json_writing":json_writing,
    "load_float":load_float,
    "load_bool":load_bool,
    "file_path_iterator":file_path_iterator,
    "aisuite_loader":aisuite_loader,
}
if enable_interpreter:
    NODE_CLASS_MAPPINGS["interpreter_tool"] = interpreter_tool
    NODE_CLASS_MAPPINGS["interpreter_function"] = interpreter_function

lang = locale.getdefaultlocale()[0]
api_keys = load_api_keys(config_path)
lang_config=api_keys.get("language")
if lang_config=="en_US" or lang_config=="zh_CN":
    lang=lang_config
if lang == "zh_CN":
    NODE_DISPLAY_NAME_MAPPINGS = {
        "LLM": "☁️API LLM通用链路",
        "LLM_local": "🖥️本地LLM通用链路",
        "LLM_api_loader": "☁️API LLM加载器",
        "easy_LLM_api_loader": "☁️简易API LLM加载器",
        # "genai_api_loader":"Gemini API LLM加载器",
        "LLM_local_loader": "🖥️本地LLM加载器",
        "easy_LLM_local_loader": "🖥️简易本地LLM加载器",
        "load_ebd": "🖥️加载词嵌入模型",
        "embeddings_function": "🖥️词向量检索",
        "load_file": "加载文件",
        "load_persona": "加载人格面具",
        "classify_persona": "分类器面具",
        "classify_function": "分类器函数",
        "classify_persona_plus": "超大分类器面具",
        "classify_function_plus": "超大分类器函数",
        "tool_combine": "工具组合",
        "tool_combine_plus": "超大工具组合",
        "time_tool": "时间工具",
        "accuweather_tool": "accuweather工具",
        "google_tool": "谷歌搜索工具",
        "bing_tool": "必应搜索工具",
        "check_web_tool": "检视网页工具",
        "file_combine": "文件组合",
        "file_combine_plus": "超大文件组合",
        "string_combine": "字符串组合",
        "string_combine_plus": "超大字符串组合",
        "start_dialog": "开始对话",
        "end_dialog": "结束对话",
        # "interpreter_tool": "代码执行工具",
        "ebd_tool": "🖥️词嵌入模型工具",
        "custom_persona": "自定义面具",
        "start_workflow": "开始工作流",
        "end_workflow": "结束工作流",
        "CLIPTextEncode_party": "CLIP文本编码器",
        "KSampler_party": "KSampler采样器",
        "VAEDecode_party": "VAEDecode解码器",
        "string_logic": "字符串逻辑",
        "show_text_party": "显示文本",
        "load_url": "加载网页内容",
        "load_file_folder": "加载文件夹",
        "api_tool": "API工具",
        "wikipedia_tool": "维基百科工具",
        "load_wikipedia": "加载维基百科",
        "workflow_transfer": "工作流中转器",
        "About_us": "关于我们",
        "workflow_tool": "工作流工具",
        "github_tool": "GitHub工具",
        "work_wechat_tool": "企业微信工具",
        "work_wechat": "发送到企业微信",
        "Dingding_tool": "钉钉工具",
        "Dingding": "发送到钉钉",
        "feishu_tool": "飞书工具",
        "feishu": "发送到飞书",
        "substring": "提取字符串",
        "openai_tts": "☁️OpenAI语音合成",
        "load_name": "加载config.ini中的模型名称",
        "omost_decode": "omost解码器",
        "omost_setting": "omost设置",
        "keyword_tool": "搜索关键词工具",
        "load_keyword": "加载关键词检索器",
        "story_json_tool": "故事JSON工具",
        "KG_json_toolkit_developer": "知识图谱JSON工具包开发者版",
        "KG_json_toolkit_user": "知识图谱JSON工具包用户版",
        "KG_csv_toolkit_developer": "知识图谱CSV工具包开发者版",
        "KG_csv_toolkit_user": "知识图谱CSV工具包用户版",
        "replace_string": "替换字符串",
        "KG_neo_toolkit_developer": "知识图谱Neo4j工具包开发者版",
        "KG_neo_toolkit_user": "知识图谱Neo4j工具包用户版",
        "translate_persona": "翻译面具",
        "load_excel": "Excel迭代器",
        "text_iterator": "文本迭代器",
        "image_iterator": "图片迭代器",
        "google_loader": "Google搜索加载器",
        "bing_loader": "Bing搜索加载器",
        "api_function": "API函数",
        "parameter_function": "参数字典函数",
        "get_string": "输入字符串",
        "parameter_combine": "参数字典组合",
        "parameter_combine_plus": "超大参数字典组合",
        "list_append": "列表追加",
        "list_append_plus": "超大列表追加",
        "list_extend": "列表扩展",
        "list_extend_plus": "超大列表扩展",
        "clear_model": "清空模型",
        "save_ebd_database": "🖥️保存向量数据库",
        "json2text": "JSON转文本",
        # "interpreter_function": "解释器函数",
        "load_img_path": "从图片路径加载",
        "img2path": "图片存至路径",
        "load_int": "加载整数",
        "none2false": "None转False",
        "bool_logic": "布尔逻辑",
        "duckduckgo_tool": "DuckDuckGo工具",
        "duckduckgo_loader": "DuckDuckGo加载器",
        "flux_persona":"flux提示词生成器面具",
        "clear_file":"清理文件",
        "workflow_transfer_v2":"工作流中转器V2",
        "str2float":"字符串转浮点数",
        "json_iterator":"JSON迭代器",
        "Lorebook":"Lorebook传说书",
        "text_writing":"文件写入",
        "str2int":"字符串转整数",
        "any2str":"任意类型转字符串",
        "start_anything":"开始任意",
        "end_anything": "结束任意",
        "json_writing": "JSON写入",
        "load_float": "加载浮点数",
        "load_bool": "加载布尔值",
        "file_path_iterator": "文件路径迭代器",
        "aisuite_loader": "☁️AISuite加载器",
    }
    if enable_interpreter:
        NODE_DISPLAY_NAME_MAPPINGS["interpreter_tool"] = "代码执行工具"
        NODE_DISPLAY_NAME_MAPPINGS["interpreter_function"] = "解释器函数"
else:
    NODE_DISPLAY_NAME_MAPPINGS = {
        "LLM": "☁️API LLM general link",
        "LLM_local": "🖥️Local LLM general link",
        "LLM_api_loader": "☁️API LLM Loader",
        "easy_LLM_api_loader": "☁️Easy API LLM Loader",
        # "genai_api_loader":"Gemini API LLM Loader",
        "LLM_local_loader": "🖥️Local LLM Loader",
        "easy_LLM_local_loader": "🖥️Easy Local LLM Loader",
        "load_ebd": "🖥️Load Embeddings",
        "embeddings_function": "🖥️Word Vector Search",
        "load_file": "Load File",
        "load_persona": "Load Persona",
        "classify_persona": "Classify Persona",
        "classify_function": "Classify Function",
        "classify_persona_plus": "Large Classify Persona",
        "classify_function_plus": "Large Classify Function",
        "tool_combine": "Tool Combine",
        "tool_combine_plus": "Large Tool Combine",
        "time_tool": "Time Tool",
        "accuweather_tool": "accuweather Tool",
        "google_tool": "Google Search Tool",
        "bing_tool": "Bing Search Tool",
        "check_web_tool": "Check Web Tool",
        "file_combine": "File Combine",
        "file_combine_plus": "Large File Combine",
        "string_combine": "String Combine",
        "string_combine_plus": "Large String Combine",
        "start_dialog": "Start Dialog",
        "end_dialog": "End Dialog",
        # "interpreter_tool": "Code Execution Tool",
        "ebd_tool": "🖥️Embeddings Tool",
        "custom_persona": "Custom Persona",
        "start_workflow": "Start Workflow",
        "end_workflow": "End Workflow",
        "CLIPTextEncode_party": "CLIP Text Encoder",
        "KSampler_party": "KSampler",
        "VAEDecode_party": "VAEDecode",
        "string_logic": "String Logic",
        "show_text_party": "Show Text",
        "load_url": "Load URL Content",
        "load_file_folder": "Load File Folder",
        "api_tool": "API Tool",
        "wikipedia_tool": "Wikipedia Tool",
        "load_wikipedia": "Load Wikipedia",
        "workflow_transfer": "Workflow Transfer",
        "About_us": "About Us",
        "workflow_tool": "Workflow Tool",
        "github_tool": "GitHub Tool",
        "work_wechat_tool": "Work WeChat Tool",
        "work_wechat": "Send to Work WeChat",
        "Dingding_tool": "Dingding Tool",
        "Dingding": "Send to Dingding",
        "feishu_tool": "Feishu Tool",
        "feishu": "Send to Feishu",
        "substring": "Extract Substring",
        "openai_tts": "☁️OpenAI TTS",
        "load_name": "Load Model Name in config.ini",
        "omost_decode": "omost Decoder",
        "omost_setting": "omost Setting",
        "keyword_tool": "Search Keyword Tool",
        "load_keyword": "Load Keyword Searcher",
        "story_json_tool": "Story JSON Tool",
        "KG_json_toolkit_developer": "KG JSON Toolkit Developer",
        "KG_json_toolkit_user": "KG JSON Toolkit User",
        "KG_csv_toolkit_developer": "KG CSV Toolkit Developer",
        "KG_csv_toolkit_user": "KG CSV Toolkit User",
        "replace_string": "Replace String",
        "KG_neo_toolkit_developer": "KG Neo4j Toolkit Developer",
        "KG_neo_toolkit_user": "KG Neo4j Toolkit User",
        "translate_persona": "Translate Persona",
        "load_excel": "Excel Iterator",
        "text_iterator": "Text Iterator",
        "image_iterator": "Image Iterator",
        "google_loader": "Google Image Loader",
        "bing_loader": "Bing Image Loader",
        "api_function": "API Function",
        "parameter_function": "Parameter Dictionary Function",
        "get_string": "Input String",
        "parameter_combine": "Parameter Dictionary Combine",
        "parameter_combine_plus": "Large Parameter Dictionary Combine",
        "list_append": "List Append",
        "list_append_plus": "Large List Append",
        "list_extend": "List Extend",
        "list_extend_plus": "Large List Extend",
        "clear_model": "Clear Model",
        "save_ebd_database": "🖥️Save Embeddings Database",
        "json2text": "JSON to Text",
        # "interpreter_function": "Interpreter Function",
        "load_img_path": "Load Image from Path",
        "img2path": "Image to Path",
        "load_int": "Load Integer",
        "none2false": "None to False",
        "bool_logic": "Boolean Logic",
        "duckduckgo_tool": "DuckDuckGo Tool",
        "duckduckgo_loader":"DuckDuckGo Loader",
        "flux_persona":"flux prompt word generator",
        "clear_file":"clear file",
        "workflow_transfer_v2": "Workflow Transfer V2",
        "str2float": "String to Float",
        "json_iterator": "JSON Iterator",
        "Lorebook":"Lore book",
        "text_writing":"File write",
        "str2int": "String to Integer",
        "any2str": "Any to String",
        "start_anything": "Start Anything",
        "end_anything": "End Anything",
        "json_writing": "JSON Writing",
        "load_float": "Load Float",
        "load_bool": "Load Boolean",
        "file_path_iterator":"File Path Iterator",
        "aisuite_loader":"☁️Aisuite Loader"
    }
    if enable_interpreter:
        NODE_DISPLAY_NAME_MAPPINGS["interpreter_tool"] = "Code Execution Tool"
        NODE_DISPLAY_NAME_MAPPINGS["interpreter_function"] = "Interpreter Function"

def load_custom_tools():
    # 获取 custom_tool 文件夹的路径
    custom_tool_dir = os.path.join(os.path.dirname(__file__), "custom_tool")

    # 找到 custom_tool 文件夹下所有的 .py 文件
    files = glob.glob(os.path.join(custom_tool_dir, "*.py"), recursive=False)

    for file in files:
        # 获取文件名（不包含扩展名）
        name = os.path.splitext(os.path.basename(file))[0]

        try:
            # 创建一个导入规范
            spec = importlib.util.spec_from_file_location(name, file)

            # 根据导入规范创建一个新的模块对象
            module = importlib.util.module_from_spec(spec)

            # 在 sys.modules 中注册这个模块
            sys.modules[name] = module

            # 执行模块的代码，实际加载模块
            spec.loader.exec_module(module)

            # 如果模块有 NODE_CLASS_MAPPINGS 属性，更新字典
            if hasattr(module, "NODE_CLASS_MAPPINGS"):
                NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))

            # 如果模块有 NODE_DISPLAY_NAME_MAPPINGS 属性，更新字典
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

            # 如果模块有 _TOOL_HOOKS 属性，将其字符串添加到 _TOOL_HOOKS 列表中
            if hasattr(module, "_TOOL_HOOKS"):
                _TOOL_HOOKS.extend(getattr(module, "_TOOL_HOOKS", []))

                # 对于每个工具名称，将同名的函数导入到全局变量中
                for tool_name in getattr(module, "_TOOL_HOOKS", []):
                    if hasattr(module, tool_name):
                        # 将函数添加到全局变量中
                        globals()[tool_name] = getattr(module, tool_name)

        except Exception as e:
            # 处理导入错误（例如，跳过文件）
            print(f"Optional node {name} import failed with error: {e}.If you don't need to use this optional node, this reminder can be ignored.")


# 调用函数来加载 custom_tool 文件夹下的模块
load_custom_tools()


if __name__ == "__main__":
    print("hello llm party!")

