import os
import hashlib
import json
import io
from PIL import Image
from datasets import load_dataset

# Load the WildVision/wildvision-bench dataset
dataset = load_dataset("WildVision/wildvision-bench", split="test")

# Define the output directory for images
output_image_dir = "/home/babychou/wv_bench_images"
os.makedirs(output_image_dir, exist_ok=True)

# Define the output JSONL file
output_jsonl_file = "/home/babychou/wv_bench.jsonl"

with open(output_jsonl_file, "w") as outfile:
    for example in dataset:
        question_id = example["question_id"]
        instruction = example["instruction"]
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
            "category": "wv_bench",
            "turns": [{"content": content}]
        }

        # Write the JSON object as a line in the JSONL file
        outfile.write(json.dumps(json_object) + "\n")
