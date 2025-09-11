import json
from tqdm import tqdm
from pathlib import Path

def process_single_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        lines = infile.readlines()
        
        for line in tqdm(lines, desc=f"Processing {input_path.name}"):
            data = json.loads(line)
            
            for message in data.get("messages", []):
                if message.get("role") == "assistant" and "content" in message:
                    message["content"] = ""
            
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

def main():
    #change this
    input_file = Path("xxq/train/stance_split_meta/test.jsonl")
    output_file = Path("xxq/train/stance_split_meta/test_infer.jsonl")
    #change end

    process_single_jsonl(input_file, output_file)

if __name__ == "__main__":
    main()