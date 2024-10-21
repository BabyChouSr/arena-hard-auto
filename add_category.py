import json
import argparse

input_file = "/home/babychou/wv_bench.jsonl"
output_file = "/home/babychou/wv_bench.jsonl"

def add_category(input_file, output_file):
    new_lines = []
    with open(input_file, 'r') as infile:
        for line in infile:
            obj = json.loads(line)
            obj['category'] = 'wv_bench'
            new_lines.append(json.dumps(obj) + '\n')

    with open(output_file, 'w') as outfile:
        outfile.writelines(new_lines)

if __name__ == '__main__':
    add_category(input_file, output_file)
