import argparse
import json
import os
from typing import List

def filter_error_lines(input_file, bad_question_ids: List[str]):
    filtered_lines = []
    with open(input_file, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            if len(bad_question_ids) > 0 and data.get("question_id", "") in bad_question_ids:
                print(f"Skipping {data['question_id']}")
                continue

            if "games" in data:
                filtered_lines.append(line)
                continue

            content = data['choices'][0]['turns'][0]['content']
            if content is None:
                print(f"Deleted empty line: {line.strip()}")
            elif "$ERROR$" not in content:
                filtered_lines.append(line)
            else:
                print(f"Deleted line: {line.strip()}")

    filtered_lines = list(set(filtered_lines))
    
    return filtered_lines

def filter_error_dir(input_dir, bad_question_ids: List[str]):
    for file in os.listdir(input_dir):
        if file.endswith('.jsonl'):
            filtered_lines = filter_error_lines(os.path.join(input_dir, file), bad_question_ids)
            with open(os.path.join(input_dir, file), 'w') as outfile:
                for line in filtered_lines:
                    outfile.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter out lines containing '$ERROR$' from a JSONL file.")
    parser.add_argument('--input_dir', help="Path to the input JSONL file")
    parser.add_argument('--bad_question_ids', nargs="+", default=[],help="Path to the bad question ids file")
    args = parser.parse_args()

    filter_error_dir(args.input_dir, args.bad_question_ids)

    print(f"Filtered content has been written back to {args.input_dir}")