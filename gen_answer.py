"""Generate answers using api endpoints.

Usage:
python gen_api_answer --parallel 32
"""
import argparse
import json
import os
import re
import time
import concurrent.futures
import multiprocessing
from dataclasses import dataclass
from multiprocessing import Process
from typing import List, Dict, Any, Optional

import tiktoken
import shortuuid
import tqdm

from add_markdown_info import count_markdown_elements, remove_pattern
from config.configs import GenAnswerConfig, EndpointInfo, EndpointsConfig
from utils import (
    load_questions,
    load_model_answers,
    make_config,
    get_endpoint,
    chat_completion_litellm,
    chat_completion_openai,
    chat_completion_anthropic,
    chat_completion_openai_azure,
    chat_completion_mistral,
    http_completion_gemini,
    chat_completion_cohere,
    chat_completion_reka,
    reorg_answer_file,
    OPENAI_MODEL_LIST,
    temperature_config,
    _content_to_openai_format,
    get_filepath
)


def get_answer(
    question: dict, model: str, endpoint_info: EndpointInfo, num_choices: int, max_tokens: int, temperature: float, answer_file: str, api_dict: dict, images_base_dir: str
):
    if question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]

    api_type = endpoint_info.api_type

    conv = []

    if endpoint_info.system_prompt:
        conv.append({"role": "system", "content": endpoint_info.system_prompt})
    elif model in OPENAI_MODEL_LIST:
        conv.append({"role": "system", "content": "You are a helpful assistant."})

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    choices = []
    for i in range(num_choices):
        turns = []
        for j in range(len(question["turns"])):
            print(question["question_id"])
            conv.append({"role": "user", "content": _content_to_openai_format(question["turns"][j]["content"], images_base_dir)})
            if api_type == "anthropic":
                output = chat_completion_anthropic(model=endpoint_info.model_name,
                                                   messages=conv,
                                                   temperature=temperature,
                                                   max_tokens=max_tokens)
            elif api_type == "mistral":
                output = chat_completion_mistral(model=endpoint_info.model_name,
                                                 messages=conv,
                                                 temperature=temperature,
                                                 max_tokens=max_tokens)
            elif api_type == "gemini":
                output = http_completion_gemini(model=endpoint_info.model_name,
                                                message=question["turns"][j]["content"],
                                                temperature=temperature,
                                                max_tokens=max_tokens)
            elif api_type == "azure":
                output = chat_completion_openai_azure(model=endpoint_info.model_name,
                                                      messages=conv,
                                                      temperature=temperature,
                                                      max_tokens=max_tokens,
                                                      api_dict=api_dict)
            elif api_type == "cohere":
                output = chat_completion_cohere(model=endpoint_info.model_name,
                                                messages=conv,
                                                temperature=temperature,
                                                max_tokens=max_tokens)
            elif api_type == "litellm":
                output = chat_completion_litellm(model=endpoint_info.model_name,
                                                 messages=conv,
                                                 temperature=temperature,
                                                 max_tokens=max_tokens)
            elif api_type == "reka":
                output = chat_completion_reka(model=endpoint_info.model_name,
                                                 messages=conv,
                                                 temperature=temperature,
                                                 max_tokens=max_tokens)
            else:
                output = chat_completion_openai(model=endpoint_info.model_name, 
                                                messages=conv, 
                                                temperature=temperature, 
                                                max_tokens=max_tokens, 
                                                api_dict=api_dict)
            conv.append({"role": "assistant", "content": output})

            turns.append({"content": output})
        choices.append({"index": i, "turns": turns})
    
    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }
    
    if len(choices) == len(turns) == 1:
        if output:
            metadata = {"token_len": len(encoding.encode(output, 
                                                     disallowed_special=()))}
            metadata = metadata | count_markdown_elements(remove_pattern(output, 
                                                                     re.compile("```([^`]*)```")),
                                                                 suffix="")
        else:
            metadata = {"token_len": 0}
        ans["conv_metadata"] = metadata

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")

def get_answer_process(questions_queue: multiprocessing.Queue):
    while True:
        item: Optional[QuestionsInfo] = questions_queue.get()
        if item is None:
            break
    
        with concurrent.futures.ThreadPoolExecutor(max_workers=item.endpoint_info.parallel) as executor:
            futures = []
            for index, question in enumerate(item.questions):
                future = executor.submit(
                    get_answer,
                    question, item.model, item.endpoint_info, item.num_choices, item.max_tokens_list[index], item.temperature, item.answer_file, item.api_dict, item.images_base_dir,
                )
                futures.append(future)

            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                future.result()

        reorg_answer_file(item.answer_file)
        print(f"Finished {item.model} for {len(item.questions)} questions")

@dataclass
class QuestionsInfo:
    endpoint_info: EndpointInfo
    questions: List[Dict[str, Any]]
    model: str
    num_choices: int
    max_tokens_list: List[int]
    temperature: float
    answer_file: str
    api_dict: Dict[str, Any]
    images_base_dir: str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting-file", type=str, default="config/gen_answer_config.yaml"
    )
    parser.add_argument(
        "--endpoint-file", type=str, default="config/api_config.yaml"
    )
    parser.add_argument(
        "--question-file", type=str, default="", help="Path to the question file that model answers to",
    )
    parser.add_argument(
        "--answers-base-dir", type=str, default = "", help="Output path that stores the model's answers",
    )
    parser.add_argument(
        "--images-base-dir", type=str, default = "", help="Path to the images that model answers to",
    )
    parser.add_argument(
        "--num-workers", type=int, default=multiprocessing.cpu_count(), help="Number of workers to generate answers",
    )
    args = parser.parse_args()

    settings = GenAnswerConfig.from_dict(make_config(args.setting_file))
    endpoints_config = EndpointsConfig.from_dict(make_config(args.endpoint_file))

    existing_answer_dir = get_filepath(args.answers_base_dir, os.path.join("data", settings.bench_name, "model_answer"))
    existing_answer = load_model_answers(existing_answer_dir)
    
    print(settings)
    print(endpoints_config)

    queue = multiprocessing.Queue()
    num_workers = min(args.num_workers, len(settings.model_list))
    for model in settings.model_list:
        assert model in endpoints_config.endpoints, f"{model} not found in {endpoints_config.endpoints.keys()}"
        endpoint_info = endpoints_config.endpoints[model]

        question_file = get_filepath(args.question_file, os.path.join("data", settings.bench_name, "question.jsonl"))
        questions = load_questions(question_file)

        answer_dir = get_filepath(args.answers_base_dir, os.path.join("data", settings.bench_name, "model_answer"))
        answer_file = os.path.join(answer_dir, f"{model}.jsonl")
        print(f"Output to {answer_file}")

        if endpoint_info.parallel:
            parallel = endpoint_info.parallel
        else:
            parallel = 1

        # We want to maximizes the number of tokens generate per answer: max_tokens = specified token # - input tokens #
        if endpoint_info.tokenizer:
            question_list = [question["turns"][0]["content"] for question in questions]
            if model in OPENAI_MODEL_LIST:
                tokenizer = tiktoken.encoding_for_model(endpoint_info.model_name)
                tokens = [tokenizer.encode(prompt) for prompt in question_list]
                max_tokens = [(settings.max_tokens- len(token) - 100) for token in tokens]
            else:
                from transformers import AutoTokenizer
                
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                tokenizer = AutoTokenizer.from_pretrained(endpoint_info.tokenizer)

                tokens = tokenizer(question_list)
                max_tokens = [(settings.max_tokens - len(prompt) - 300) for prompt in tokens["input_ids"]]
        else:
            max_tokens = [settings.max_tokens] * len(questions)
        
        questions_without_existing_answer = []
        max_tokens_for_questions_without_existing_answer = []
        count = 0
        for index, question in enumerate(questions):
            if model in existing_answer and question["question_id"] in existing_answer[model]:
                count += 1
                continue
            questions_without_existing_answer.append(question)
            max_tokens_for_questions_without_existing_answer.append(max_tokens[index])
        if count > 0:
            print(f"{count} number of existing answers")

        questions_info = QuestionsInfo(endpoint_info, questions_without_existing_answer, model, settings.num_choices, max_tokens_for_questions_without_existing_answer, settings.temperature, answer_file, get_endpoint(endpoint_info.endpoints), args.images_base_dir)
        queue.put(questions_info)

    for _ in range(num_workers):
        queue.put(None)

    workers = []
    for _ in range(num_workers):
        worker = multiprocessing.Process(target=get_answer_process, args=(queue,))
        worker.start()
        workers.append(worker)

    for worker in workers:
        worker.join()