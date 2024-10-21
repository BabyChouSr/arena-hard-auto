import json
import argparse

def change_content_in_jsonl(input_file):
    with open(input_file, 'r') as infile:
        data_list = [json.loads(line) for line in infile]

    for data in data_list:
        for turn in data.get('turns', []):
            if 'content' in turn and isinstance(turn['content'], list) and len(turn['content']) > 0:
                turn['content'][0] = "What is in this image?"

    with open(input_file, 'w') as outfile:
        for data in data_list:
            outfile.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Change the first element in the content list to "What is in this image?".')
    parser.add_argument('--file', type=str, required=True, help='Path to the JSONL file to modify in place.')
    args = parser.parse_args()
    change_content_in_jsonl(args.file)
