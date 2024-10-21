import argparse
import json
import random

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def save_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            question_id = item["question_id"]
            category = "arena-hard-vl-random"
            turns = [
                {
                    "content": item["conversation_a"][0]["content"]
                }
            ]
            res = {
                "question_id": question_id,
                "category": category,
                "turns": turns
            }
            json.dump(res, file)
            file.write('\n')

def random_filter(input_file, output_file, num_rows, seed, language):
    random.seed(seed)
    data = load_jsonl(input_file)

    data_filtered_by_language = [d for d in data if d["language"] == language]
    print(f"Number of rows in {language}: {len(data_filtered_by_language)}")
    
    if num_rows >= len(data_filtered_by_language):
        filtered_data = data_filtered_by_language
    else:
        filtered_data = random.sample(data_filtered_by_language, num_rows)
    
    save_jsonl(filtered_data, output_file)
    print(f"Randomly selected {len(filtered_data)} rows and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Randomly filter rows from a JSONL file.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file")
    parser.add_argument("--num_rows", type=int, default=500, help="Number of rows to randomly select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--language", type=str, default="English", help="Language")
    
    args = parser.parse_args()
    
    random_filter(args.input_file, args.output_file, args.num_rows, args.seed, args.language)

if __name__ == "__main__":
    main()
