# 导入所需的库
#api调用代码
import os
from openai import OpenAI
import json
import random
from collections import defaultdict # <--- 新增导入 defaultdict

# --- 1. 配置您的API信息 ---
API_KEY = "sk-b386510df5614cb1bdde83e71f65e7b0"
MODEL_NAME = "qwen-plus"

# 初始化API客户端
try:
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
except Exception as e:
    print(f"初始化API客户端时出错: {e}")
    client = None

# --- 【核心修改】加载JSONL文件并按类别分组 ---
def load_and_group_examples(file_path: str) -> dict:
    """从JSONL文件中加载所有对话示例，并按助手的回答（类别）进行分组。"""
    examples_by_class = defaultdict(list)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                user_message = data['messages'][1]
                assistant_message = data['messages'][2]
                
                # 获取类别标签
                label = assistant_message.get("content")
                if label in ['0', '1', '2', '3']:
                    # 将用户问题和助手回答作为一个整体存起来
                    examples_by_class[label].append({
                        "user": user_message,
                        "assistant": assistant_message
                    })
        
        print("示例加载和分组完成。各类别样本数量：")
        for label, items in examples_by_class.items():
            print(f"  - 等级 {label}: {len(items)} 个")
        return examples_by_class
        
    except FileNotFoundError:
        print(f"警告：示例文件 {file_path} 未找到，将使用默认的硬编码示例。")
        return {}
    except Exception as e:
        print(f"加载示例文件时出错: {e}")
        return {}

# 在这里指定您的JSONL文件路径
# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
JSONL_FILE_PATH = "/path/to/your/output_finetune_data.jsonl" 
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ALL_EXAMPLES_BY_CLASS = load_and_group_examples(JSONL_FILE_PATH)


# --- 2. 定义核心的分类函数 (已修改) ---
def get_sensitivity_level(topic: str) -> str:
    """调用LLM API根据给定的话题判断其事件敏感度等级。"""
    if not client:
        return "错误：API客户端未成功初始化。"

    # a. 系统角色指令 (不变)
    system_prompt = """你是一个事件敏感度判断助手。事件敏感度主要分为下面四个等级：
0：娱乐新闻，科技新闻等
1：降低社会价值观：对公司企业、各类产品和社会名人等公共领域进行批评、指责的观点、陈述内容反映不符合主流价值观的思想倾向，倾向社会阴暗面的内容。
2：危害社会安全：牵涉针对个人、人群、地域与非公务组织机构的攻击性事件或观点等的内容。
3：危害国家安全：牵涉国家、政府机关、政治制度、政策法规、政治人物、恐怖主义等的内容。
你需要根据输入内容输出敏感度等级，只能输出数字 0/1/2/3，不要输出任何其他内容。"""

    # --- 【核心修改】 ---
    # b. 从每个类别中各选一个例子，而不是完全随机
    few_shot_examples = []
    if ALL_EXAMPLES_BY_CLASS:
        for label in ['0', '1', '2', '3']:
            if label in ALL_EXAMPLES_BY_CLASS and ALL_EXAMPLES_BY_CLASS[label]:
                # 从该类别的所有例子中随机选择一个
                chosen_example = random.choice(ALL_EXAMPLES_BY_CLASS[label])
                few_shot_examples.append(chosen_example['user'])
                few_shot_examples.append(chosen_example['assistant'])
    
    # 如果加载文件失败或文件为空，退回到原来的硬编码示例
    if not few_shot_examples:
        few_shot_examples = [
            {"role": "user", "content": "话题名称：22岁冠心病女生不想做支架狂炫中药"},
            {"role": "assistant", "content": "0"},
            {"role": "user", "content": "话题名称：8人被终身禁入稻城亚丁景区"},
            {"role": "assistant", "content": "0"}
        ]
    
    # c. 最终的用户请求 (不变)
    new_user_request = {
        "role": "user",
        "content": f"话题名称：{topic}"
    }
    
    # 组合成完整的对话消息列表
    messages_for_api = [
        {"role": "system", "content": system_prompt}
    ] + few_shot_examples + [new_user_request]

    # --- 调用API (不变) ---
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages_for_api,
            temperature=0.0,
            top_p=0.1
        )
        result = completion.choices[0].message.content.strip()
        if result in ['0', '1', '2', '3']:
            return result
        else:
            return f"模型返回了意外内容: '{result}'"
    except Exception as e:
        return f"调用API时发生错误: {e}"

# --- 3. 交互式命令行界面 (保持不变) ---
if __name__ == "__main__":
    print("🤖 事件敏感度判断助手已启动 (v3.0 - 分层抽样版)")
    print("   - 输入一个话题，助手将返回敏感度等级 (0/1/2/3)")
    print("   - 输入 'quit' 或 'exit' 退出程序\n")
    while True:
        user_topic = input("请输入话题名称 > ")
        if user_topic.lower() in ["quit", "exit", "q"]:
            print("👋 感谢使用，再见！")
            break
        if not user_topic:
            print("⚠️ 输入不能为空，请重新输入。")
            continue
        print("正在判断中...")
        level = get_sensitivity_level(user_topic)
        print(f"敏感度等级: {level}\n")