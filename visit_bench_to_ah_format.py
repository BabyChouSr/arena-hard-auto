import os
import hashlib
import json
import io
import time
import tiktoken
from PIL import Image
from datasets import load_dataset

# Load the WildVision/wildvision-bench dataset
dataset = load_dataset("mlfoundations/VisIT-Bench", split="test")

# Define the output directory for images
output_image_dir = "/home/babychou/visit_bench_images"
os.makedirs(output_image_dir, exist_ok=True)

# Define the output JSONL file
output_jsonl_file = "/home/babychou/visit_bench/bench.jsonl"

gpt4_reference_answer_jsonl_file = "/home/babychou/visit_bench/model_answer/gpt-4-ref.jsonl"

os.makedirs(os.path.dirname(output_jsonl_file), exist_ok=True)
os.makedirs(os.path.dirname(gpt4_reference_answer_jsonl_file), exist_ok=True)

tokenizer = tiktoken.encoding_for_model("gpt-4-turbo")

with open(output_jsonl_file, "w") as outfile, open(gpt4_reference_answer_jsonl_file, "w") as gpt4_outfile:
    for example in dataset:
        if not example["human_ratings_gpt4_correct"]:
            continue
        
        instruction = example["instruction"]
        question_id = hashlib.md5(instruction.encode()).hexdigest()
        image_bytes = example["image"]


        img_byte_arr = io.BytesIO()
        image_bytes.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        # Compute MD5 hash of the image bytes
        md5_hash = hashlib.md5(img_byte_arr).hexdigest()

        # Save the image to the output directory
        image_filename = f"{md5_hash}.png"
        image_path = os.path.join(output_image_dir, image_filename)
        pil_image = Image.open(io.BytesIO(img_byte_arr))
        pil_image.save(image_path)

        # Create the content list
        content = [instruction, [md5_hash]]

        # Prepare the JSON object
        json_object = {
            "question_id": question_id,
            "category": "visit_bench",
            "turns": [{"content": content}]
        }

        # Write the JSON object as a line in the JSONL file
        outfile.write(json.dumps(json_object) + "\n")

        gpt_4_ref_answer = example["gpt4_prediction"]
        answer_id = hashlib.md5(gpt_4_ref_answer.encode()).hexdigest()
        tokens = tokenizer.encode(gpt_4_ref_answer)
        gpt4_outfile.write(json.dumps({
            "question_id": question_id,
            "answer_id": answer_id,
            "model_id": "gpt-4",
            "choices": [{"index": 0, "turns": [{"content": gpt_4_ref_answer}]}],
            "tstamp": time.time(),
            "conv_metadata": {
                "token_len": len(tokens),
                "header_count": {
                    "h1": 0,
                    "h2": 0,
                    "h3": 0,
                    "h4": 0,
                    "h5": 0,
                    "h6": 0
                }
            }
        }) + "\n")
        