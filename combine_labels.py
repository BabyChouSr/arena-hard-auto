import argparse
import json

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def combine_labels(file1, file2, output_file):
    data1 = load_jsonl(file1)
    data2 = load_jsonl(file2)

    data2_dict = {entry['question_id']: entry for entry in data2}

    combined_data = []
    for entry in data1:
        question_id = entry['question_id']
        if question_id in data2_dict:
            entry['category_tag'].update(data2_dict[question_id]['category_tag'])
        combined_data.append(entry)

    save_jsonl(combined_data, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine labels from two JSONL files.")
    parser.add_argument("--file1", type=str, help="Path to the first JSONL file.")
    parser.add_argument("--file2", type=str, help="Path to the second JSONL file.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file.")
    args = parser.parse_args()

    combine_labels(args.file1, args.file2, args.output_file)
