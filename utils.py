import base64
import os
import json
import time
import yaml
import random
import requests
from typing import List, Dict, Tuple, Union

from typing import Optional
from glob import glob

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0613-verbose",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
)


temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}


def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(
        endpoint_list
    )[0]
    return api_dict


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None):
    import openai
    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                )
            output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except KeyError:
            print(type(e), e)
            break
    
    return output


def chat_completion_openai_azure(model, messages, temperature, max_tokens, api_dict=None):
    import openai
    from openai import AzureOpenAI

    api_base = api_dict["api_base"]
    client = AzureOpenAI(
        azure_endpoint = api_base,
        api_key= api_dict["api_key"],
        api_version=api_dict["api_version"],
        timeout=240,
        max_retries=2
    )

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42,
            )
            output = response.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(type(e), e)
            break

    return output


def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None):
    import anthropic

    if api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output


def chat_completion_mistral(model, messages, temperature, max_tokens):
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralException

    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)

    prompts = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            chat_response = client.chat(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = chat_response.choices[0].message.content
            break
        except MistralException as e:
            print(type(e), e)
            break

    return output


def http_completion_gemini(model, message, temperature, max_tokens):
    api_key = os.environ["GEMINI_API_KEY"]
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    output = API_ERROR_OUTPUT
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            json={
                "contents": [{
                    "parts":[
                        {"text": message}
                    ]
                }],
                "safetySettings": safety_settings,
                "generationConfig":{
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                }
            },
        )
    except Exception as e:
        print(f"**API REQUEST ERROR** Reason: {e}.")

    if response.status_code != 200:
        print(f"**API REQUEST ERROR** Reason: status code {response.status_code}.")

    output = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    return output
    


def chat_completion_cohere(model, messages, temperature, max_tokens):
    import cohere

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    assert len(messages) > 0

    template_map = {"system":"SYSTEM",
                    "assistant":"CHATBOT",
                    "user":"USER"}

    assert messages[-1]["role"] == "user"
    prompt = messages[-1]["content"]

    if len(messages) > 1:
        history = []
        for message in messages[:-1]:
            history.append({"role":template_map[message["role"]], "message":message["content"]})
    else:
        history = None

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = co.chat(
                message=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=history,
            )
            output = response.text
            break
        except cohere.core.api_error.ApiError as e:
            print(type(e), e)
            raise
        except Exception as e:
            print(type(e), e)
            break
    
    return output

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def resize_image_and_encode_image(image_path: str, max_image_size_mb: float):
    import math
    from io import BytesIO
    from PIL import Image

    image = Image.open(image_path)
    image_format = "png"
    max_hw, min_hw = max(image.size), min(image.size)
    aspect_ratio = max_hw / min_hw
    max_len, min_len = 1024, 1024
    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
    longest_edge = int(shortest_edge * aspect_ratio)
    W, H = image.size
    if longest_edge != max(image.size):
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))

    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    if max_image_size_mb:
        target_size_bytes = max_image_size_mb * 1024 * 1024

        current_size_bytes = image_bytes.tell()
        if current_size_bytes > target_size_bytes:
            resize_factor = (target_size_bytes / current_size_bytes) ** 0.5
            new_width = math.floor(image.width * resize_factor)
            new_height = math.floor(image.height * resize_factor)
            image = image.resize((new_width, new_height))

            image_bytes = BytesIO()
            image.save(image_bytes, format="PNG")
            current_size_bytes = image_bytes.tell()

        image_bytes.seek(0)

    return base64.b64encode(image_bytes.read()).decode('utf-8')

def get_image_base64_str(images_base_dir: str, image_hash: str):
    max_image_size_mb = 5 / 1.5 # Anthropic's max image size is 5MB in base64 encoded format.
    try: # Use PNG format
        image_path = os.path.join(images_base_dir, f"{image_hash}.png")
        return resize_image_and_encode_image(image_path, max_image_size_mb)
    except Exception as e:
        print(f"Error getting image {image_hash}: {e}")
        try: # Use JPG format
            image_path = f"/home/babychou/arena-hard-70k-sample-images/{image_hash}.jpg"
            return resize_image_and_encode_image(image_path, max_image_size_mb)
        except Exception as e:
            print(f"Error getting JPG image {image_hash}: {e}")
            try:
                # Make a subprocess call to gsutil
                import subprocess
                cmd = f"gsutil cp gs://arena_user_content/serve_images/{image_hash}.jpg /home/babychou/arena-hard-70k-sample-images"
                subprocess.run(cmd, shell=True, check=True)
                
                # Try to resize and encode the image again
                image_path = f"/home/babychou/arena-hard-70k-sample-images/{image_hash}.jpg"
                return resize_image_and_encode_image(image_path, max_image_size_mb)
            except Exception as sub_e:
                print(f"Error in gsutil subprocess or subsequent encoding: {sub_e}")
                raise sub_e

def _content_to_openai_format(content: Union[str, Tuple[str, List[Dict[str, str]]]], images_base_dir: str) -> Union[str, List[Dict[str, str]]]:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        content_in_openai_format = []
        text, images = content
        content_in_openai_format.append({"type": "text", "text": text})
        for image in images:
            image_base64_str = get_image_base64_str(images_base_dir, image)
            content_in_openai_format.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64_str}"}})
        return content_in_openai_format
    else:
        raise ValueError(f"Unknown content type: {type(content)}")

def chat_completion_litellm(model, messages, temperature, max_tokens):
    import litellm

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            if "gemini" in model:
                extra_kwargs = {
                    "safety_settings": [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE"
                        },
                    ]
                }
            else:
                extra_kwargs = {}
            response = litellm.completion(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, **extra_kwargs)
            output = response["choices"][0]["message"]["content"]
            break
        except Exception as e:
            print(e)
            time.sleep(60)
    
    return output

def chat_completion_reka(model, messages, temperature, max_tokens):
    from reka.client import Reka
    from reka import ChatMessage, TypedMediaContent, TypedText
    import os

    client = Reka()

    # Convert messages in OPENAI to Reka format
    reka_messages = []
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if isinstance(content, list):
            content_reka = []
            for part in content:
                if part["type"] == "text":
                    content_reka.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_url":
                    content_reka.append({"type": "image_url", "image_url": part["image_url"]["url"]})
                else:
                    raise ValueError(f"Unknown content type: {part['type']}")
            reka_messages.append({"content": content_reka, "role": "user"})
        else:
            reka_messages.append({"content": content, "role": role})

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.create(
                model=model,
                messages=reka_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            output = response.responses[0].message.content
            break
        except Exception as e:
            print(e)
    
    return output
    

def get_filepath(args_filepath: str, default_filepath: str):
    if args_filepath:
        return args_filepath
    else:
        return default_filepath

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])
