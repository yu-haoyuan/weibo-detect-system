import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

# --- 1. 定义常量 ---

# 定义系统角色的提示语内容
SYSTEM_PROMPT = """你是一个立场倾向分析助手。你需要根据用户提供的话题和微博内容，判断并给出该微博的立场倾向。立场倾向只分为三类：-1（反对或强烈反对）、0（中立）、1（支持或强烈支持）。你只能输出 -1、0、1 三个数字中的一个，不要输出其他任何内容。"""

# 定义用户角色的提示语模板
USER_PROMPT_TEMPLATE = """话题：{topic}
微博内容：{content}
请判断该微博的立场倾向。"""


# --- 2. 核心处理函数 ---
def convert_csv_to_stance_jsonl(input_csv_path: str, output_jsonl_path: str):
    """
    读取CSV文件，提取'话题'和'微博正文'，并生成用于微调模型处理的立场倾向JSONL文件。
    
    Args:
        input_csv_path (str): 输入的CSV文件路径.
        output_jsonl_path (str): 输出的JSONL文件路径.
    """
    input_path = Path(input_csv_path)
    output_path = Path(output_jsonl_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查输入文件是否存在
    if not input_path.is_file():
        print(f"错误：输入文件不存在 -> {input_path}")
        return

    # 使用Pandas读取CSV，更稳健地处理各种格式问题
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    # 使用 with 语句确保输出文件能被正确关闭
    with output_path.open("w", encoding="utf-8") as outfile:
        
        print(f"共找到 {len(df)} 条记录，开始转换为立场倾向JSONL...")
        
        # 使用 tqdm 创建进度条
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="正在转换CSV为立场JSONL"):
            
            # 安全地提取“话题”和“微博正文”列
            topic = row.get('话题', '')
            topic = "" if pd.isna(topic) else str(topic)

            content = row.get('微博正文', '')
            content = "" if pd.isna(content) else str(content)
            
            # 如果微博正文为空，则跳过该行
            if not content.strip():
                continue

            # 格式化用户输入
            user_content = USER_PROMPT_TEMPLATE.format(topic=topic, content=content)
            
            # 构建 messages 列表
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                },
                # assistant的内容为空，等待模型填充
                {
                    "role": "assistant",
                    "content": "" # 留空字符串而不是None，以兼容部分模型训练框架
                }
            ]
            # 构建包含元数据（时间戳）的记录
            timestamp = row.get('发布时间', '') # 获取时间戳
            jsonl_record = {
                "messages": messages,
                "metadata": {"timestamp": timestamp} 
            }
            outfile.write(json.dumps(jsonl_record, ensure_ascii=False) + '\n')
            # # 构建完整的JSONL记录
            # jsonl_record = {"messages": messages}
            
            # # 将记录写入文件，并添加换行符
            # outfile.write(json.dumps(jsonl_record, ensure_ascii=False) + '\n')
    
    print(f"处理完成！立场倾向JSONL已保存至: {output_path}")
