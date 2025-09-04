# 导入所需的库
from openai import OpenAI
import json
import random
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import csv # 确保导入csv库

# --- 内部辅助函数 ---

def _load_and_group_examples(file_path: str) -> dict:
    """
    从CSV或JSONL文件中加载所有对话示例，并按助手的回答（类别）进行分组。
    这是一个内部函数，为主组件服务。
    """
    examples_by_class = defaultdict(list)
    path_obj = Path(file_path)

    if not path_obj.exists():
        print(f"警告：示例文件 {file_path} 未找到，将使用默认的硬编码示例。")
        return {}
        
    try:
        # 使用 'utf-8-sig' 来自动处理BOM
        with path_obj.open('r', encoding='utf-8-sig', newline='') as f:
            # 根据文件扩展名选择不同的解析方式
            if file_path.lower().endswith('.csv'):
                reader = csv.DictReader(f)
                for row in reader:
                    # 从CSV行中提取，假设列名为 '文件名' 和 '敏感程度'
                    user_content = f"话题名称：{row.get('文件名', '')}"
                    label = row.get('敏感程度')

                    if user_content and label in ['0', '1', '2', '3']:
                        examples_by_class[label].append({
                            "user": {"role": "user", "content": user_content},
                            "assistant": {"role": "assistant", "content": label}
                        })

            elif file_path.lower().endswith('.jsonl'):
                for line in f:
                    data = json.loads(line)
                    messages = data.get('messages', [])
                    if len(messages) >= 3:
                        user_message = messages[1]
                        assistant_message = messages[2]
                        label = assistant_message.get("content")
                        if label in ['0', '1', '2', '3']:
                            examples_by_class[label].append({
                                "user": user_message,
                                "assistant": assistant_message
                            })
            else:
                print(f"错误：不支持的示例文件类型 -> {file_path}")
                return {}

        print("API few-shot 示例加载和分组完成。各类别样本数量：")
        for label, items in examples_by_class.items():
            print(f"  - 等级 {label}: {len(items)} 个")
        return examples_by_class
        
    except Exception as e:
        print(f"加载示例文件时出错: {e}")
        return {}

def _get_sensitivity_level(topic: str, client: OpenAI, model_name: str, all_examples: dict) -> str:
    """内部辅助函数，构建prompt并调用LLM API判断敏感度等级。"""
    system_prompt = """你是一个事件敏感度判断助手。事件敏感度主要分为下面四个等级：
0：娱乐新闻，科技新闻等
1：降低社会价值观：对公司企业、各类产品和社会名人等公共领域进行批评、指责的观点、陈述内容反映不符合主流价值观的思想倾向，倾向社会阴暗面的内容。
2：危害社会安全：牵涉针对个人、人群、地域与非公务组织机构的攻击性事件或观点等的内容。
3：危害国家安全：牵涉国家、政府机关、政治制度、政策法规、政治人物、恐怖主义等的内容。
你需要根据输入内容输出敏感度等级，只能输出数字 0/1/2/3，不要输出任何其他内容。"""

    few_shot_examples = []
    if all_examples:
        for label in sorted(all_examples.keys()): # sorted保证顺序
            if all_examples[label]:
                chosen_example = random.choice(all_examples[label])
                few_shot_examples.append(chosen_example['user'])
                few_shot_examples.append(chosen_example['assistant'])
    
    if not few_shot_examples:
        print("警告：未使用few-shot示例，可能影响准确度。")
        # 如果加载文件失败或文件为空，退回到硬编码示例
        few_shot_examples = [
            {"role": "user", "content": "话题名称：22岁冠心病女生不想做支架狂炫中药"},
            {"role": "assistant", "content": "0"},
            {"role": "user", "content": "话题名称：8人被终身禁入稻城亚丁景区"},
            {"role": "assistant", "content": "0"}
        ]

    new_user_request = {"role": "user", "content": f"话题名称：{topic}"}
    messages_for_api = [{"role": "system", "content": system_prompt}] + few_shot_examples + [new_user_request]

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages_for_api,
            temperature=0.0,
        )
        result = completion.choices[0].message.content.strip()
        if result in ['0', '1', '2', '3']:
            return result
        else:
            print(f"警告：模型返回了意外内容 -> '{result}'")
            return "" # 返回空字符串表示失败
    except Exception as e:
        print(f"调用API时发生错误: {e}")
        return "" # 返回空字符串表示失败

# --- 主要组件函数 ---

def process_with_api_model(
    sensitivity_jsonl_path: str, 
    output_jsonl_path: str, 
    api_key: str, 
    base_url: str, 
    model_name: str,
    few_shot_examples_path: str
):
    """
    加载输入JSONL，通过调用外部API模型进行推理，并将结果写入新的JSONL文件。
    """
    print(f"开始使用API模型 '{model_name}' 进行推理...")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"初始化API客户端时出错: {e}")
        return None

    all_examples = _load_and_group_examples(few_shot_examples_path)

    input_path = Path(sensitivity_jsonl_path)
    output_path = Path(output_jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        lines = infile.readlines()
        for line in tqdm(lines, desc="正在运行API模型推理"):
            data = json.loads(line)
            messages = data.get("messages", [])
            
            if len(messages) < 2 or messages[1].get("role") != "user":
                continue

            user_content = messages[1].get("content", "")
            topic_match = re.match(r"话题名称：(.*)", user_content)
            if not topic_match:
                continue
            
            topic = topic_match.group(1).strip()
            prediction = _get_sensitivity_level(topic, client, model_name, all_examples)

            for msg in messages:
                if msg.get("role") == "assistant":
                    msg["content"] = prediction
            
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"API模型处理完成，输出文件: {output_path}")
    return str(output_path)

