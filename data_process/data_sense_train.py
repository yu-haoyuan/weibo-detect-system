import json
from pathlib import Path
from tqdm import tqdm

SYSTEM_PROMPT_CONTENT = (
    "你是一个事件敏感度判断助手。事件敏感度主要分为下面四个等级：\n"
    "0：娱乐新闻，科技新闻等\n"
    "1：降低社会价值观：对公司企业、各类产品和社会名人等公共领域进行批评、指责的观点、陈述内容反映不符合主流价值观的思想倾向，倾向社会阴暗面的内容。\n"
    "2：危害社会安全：牵涉针对个人、人群、地域与非公务组织机构的攻击性事件或观点等的内容。\n"
    "3：危害国家安全：牵涉国家、政府机关、政治制度、政策法规、政治人物、恐怖主义等的内容。\n"
    "你需要根据输入内容输出敏感度等级，只能输出数字 0/1/2/3，不要输出任何其他内容。"
)

USER_PROMPT_TEMPLATE = "话题名称：{topic}"

def process_single_csv_topic(file_path):
    return file_path.stem

def process_inference_data(input_dir_path, output_jsonl_path):
    input_dir = Path(input_dir_path)
    output_path = Path(output_jsonl_path)
    if not input_dir.is_dir():
        return
    csv_files = sorted(list(input_dir.glob("*.csv")))
    if not csv_files:
        return
    with output_path.open("w", encoding="utf-8") as outfile:
        for file_path in tqdm(csv_files):
            topic = process_single_csv_topic(file_path)
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_CONTENT
                },
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        topic=topic
                    )
                },
                {
                    "role": "assistant",
                    "content": None
                }
            ]
            jsonl_record = {"messages": messages}
            outfile.write(json.dumps(jsonl_record, ensure_ascii=False) + '\n')

def main():
    #change this
    input_dir_path = "/Users/xiaobu/Desktop/repos/xxq/train/立场检测_train"
    output_jsonl_path = "/Users/xiaobu/Desktop/repos/xxq/data_process/敏感度判断_train/sense_detec_data_inference.jsonl"
    #change end
    output_dir = Path(output_jsonl_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    process_inference_data(input_dir_path, output_jsonl_path)

if __name__ == "__main__":
    main()