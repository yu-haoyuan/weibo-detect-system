import json
import random
from pathlib import Path
from tqdm import tqdm

def split_jsonl_file(input_path, output_dir, train_ratio=0.8, test_ratio=0.1, valid_ratio=0.1):
    """
    """
    input_file = Path(input_path)
    output_directory = Path(output_dir)

    output_directory.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print("Input file not found.")
        return

    with input_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    random.shuffle(lines)

    total_lines = len(lines)
    train_end = int(total_lines * train_ratio)
    test_end = train_end + int(total_lines * test_ratio)

    train_data = lines[:train_end]
    test_data = lines[train_end:test_end]
    valid_data = lines[test_end:]

    print("Writing train.jsonl...")
    with (output_directory / "train.jsonl").open("w", encoding="utf-8") as train_file:
        for line in tqdm(train_data):
            train_file.write(line)
    
    print("Writing test.jsonl...")
    with (output_directory / "test.jsonl").open("w", encoding="utf-8") as test_file:
        for line in tqdm(test_data):
            test_file.write(line)

    print("Writing valid.jsonl...")
    with (output_directory / "valid.jsonl").open("w", encoding="utf-8") as valid_file:
        for line in tqdm(valid_data):
            valid_file.write(line)
    
    print("Splitting complete.")

def main():
    # change this
    input_file_path = "xxq/train/data_stance_train.jsonl"
    output_directory = "xxq/train/stance_split_meta"
    # change end

    split_jsonl_file(input_file_path, output_directory)

if __name__ == "__main__":
    main()