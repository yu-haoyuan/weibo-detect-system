import csv
import json
from pathlib import Path
from tqdm import tqdm

# 新的系统提示，三分类 -1, 0, 1
SYSTEM_PROMPT_CONTENT = (
    "你是一个立场倾向分析助手。"
    "你需要根据用户提供的话题和微博内容，判断并给出该微博的立场倾向。"
    "立场倾向只分为三类：-1（反对或强烈反对）、0（中立）、1（支持或强烈支持）。"
    "你只能输出 -1、0、1 三个数字中的一个，不要输出其他任何内容。"
)

USER_PROMPT_TEMPLATE = "话题：{topic}\n微博内容：{weibo_text}\n请判断该微博的立场倾向。"

# 标签映射字典
STANCE_MAP = {
    "强烈支持": "1",
    "支持": "1",
    "中立": "0",
    "反对": "-1",
    "强烈反对": "-1"
}

def validate_row(row):
    if len(row) != 7:
        return False
    return True

def process_csv_files(input_dir_path, output_jsonl_path):
    input_dir = Path(input_dir_path)
    output_path = Path(output_jsonl_path)
    
    if not input_dir.is_dir():
        print("Input directory not found.")
        return

    csv_files = sorted(list(input_dir.glob("*.csv")))
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    with output_path.open("w", encoding="utf-8") as outfile:
        for file_path in tqdm(csv_files):
            topic = file_path.stem
            with file_path.open("r", encoding="utf-8") as infile:
                csv_reader = csv.reader(infile)
                next(csv_reader)
                
                for row in csv_reader:
                    if validate_row(row):
                        stance_raw = row[6].strip()
                        stance_mapped = STANCE_MAP.get(stance_raw, None)
                        
                        if stance_mapped is None:
                            continue
                        
                        messages = [
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT_CONTENT
                            },
                            {
                                "role": "user",
                                "content": USER_PROMPT_TEMPLATE.format(
                                    topic=topic,
                                    weibo_text=row[4].strip()
                                )
                            },
                            {
                                "role": "assistant",
                                "content": stance_mapped
                            }
                        ]
                        
                        jsonl_record = {"messages": messages}
                        outfile.write(json.dumps(jsonl_record, ensure_ascii=False) + '\n')

def main():
    # change this
    input_dir_path = "/Users/xiaobu/Desktop/repos/xxq/train/立场检测_train"
    output_jsonl_path = "/Users/xiaobu/Desktop/repos/xxq/data_process/立场检测_train/data_stance_train.jsonl"
    # change end

    output_dir = Path(output_jsonl_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    process_csv_files(input_dir_path, output_jsonl_path)

if __name__ == "__main__":
    main()