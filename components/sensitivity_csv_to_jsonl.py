import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

# --- 1. 定义常量 ---

# 定义系统角色的提示语内容
SYSTEM_PROMPT = (
    "你是一个事件敏感度判断助手。事件敏感度主要分为下面四个等级：\n"
    "0：娱乐新闻，科技新闻等\n"
    "1：降低社会价值观：对公司企业、各类产品和社会名人等公共领域进行批评、指责的观点、陈述内容反映不符合主流价值观的思想倾向，倾向社会阴暗面的内容。\n"
    "2：危害社会安全：牵涉针对个人、人群、地域与非公务组织机构的攻击性事件或观点等的内容。\n"
    "3：危害国家安全：牵涉国家、政府机关、政治制度、政策法规、政治人物、恐怖主义等的内容。\n"
    "你需要根据输入内容输出敏感度等级，只能输出数字 0/1/2/3，不要输出任何其他内容。"
)

# 定义用户角色的提示语模板
USER_PROMPT_TEMPLATE = "话题名称：{topic}"


# --- 2. 核心处理函数 ---
def convert_csv_to_sensitivity_jsonl(input_csv_path: str, output_jsonl_path: str):
    """
    读取CSV文件，并将该CSV的文件名作为话题，为整个文件生成一个JSONL条目。
    
    Args:
        input_csv_path (str): 输入的CSV文件路径.
        output_jsonl_path (str): 输出的JSONL文件路径.
    """
    input_path = Path(input_csv_path)
    output_path = Path(output_jsonl_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        print(f"错误：输入文件不存在 -> {input_path}")
        return

    # 从输入文件的路径中提取不带扩展名的文件名作为话题
    topic = input_path.stem
    print(f"已将CSV文件名 '{topic}' 作为本次处理的统一话题。")

    # --- 【核心修改】 ---
    # 由于一个文件只生成一个条目，我们不再需要读取和遍历CSV的内容
    # 只需打开输出文件并写入单个记录
        
    with output_path.open("w", encoding="utf-8") as outfile:
        print(f"正在为文件 '{input_path.name}' 创建单个JSONL条目...")
        
        # 构建 messages 列表
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                # 直接使用从文件名中提取的topic
                "content": USER_PROMPT_TEMPLATE.format(topic=topic)
            },
            # assistant的内容留空，等待API模型填充
            {
                "role": "assistant",
                "content": "" 
            }
        ]
        
        # 构建完整的JSONL记录
        jsonl_record = {"messages": messages}
        
        # 将这唯一的一条记录写入文件
        outfile.write(json.dumps(jsonl_record, ensure_ascii=False) + '\n')
            
    print(f"处理完成！单个JSONL条目已保存至: {output_path}")

