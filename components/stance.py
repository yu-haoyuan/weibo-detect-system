import json
import re
from tqdm import tqdm
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def _get_prediction(model, tokenizer, messages):
    """
    内部辅助函数，用于对单条 message 数据进行推理。
    """
    # 准备模型的聊天模板输入
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # 使用 torch.no_grad() 以节省显存并加速
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=10,       # 限制最大生成长度，对于分类任务足够
            do_sample=False,         # 使用贪心解码，确保结果一致
            eos_token_id=tokenizer.eos_token_id
        )

    # 解码模型输出，并跳过特殊token
    decoded_output = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    # 使用正则表达式从输出中提取数字（-1, 0, 1）
    match = re.search(r"(-?\d)", decoded_output)
    
    if match:
        return match.group(1)
    else:
        # 如果模型没有输出预期的数字，返回None
        print(f"警告：模型输出未能匹配到数字 -> '{decoded_output}'")
        return None

def process_with_finetuned_model(stance_jsonl_path: str, output_jsonl_path: str, model_path: str):
    """
    加载一个本地的微调模型，对输入的JSONL文件进行推理，并将结果写入新的JSONL文件。

    Args:
        stance_jsonl_path (str): 包含立场倾向任务的输入JSONL文件路径。
        output_jsonl_path (str): 保存推理结果的输出JSONL文件路径。
        model_path (str): 本地微调模型的文件夹路径。
    """
    print(f"开始使用微调模型 '{model_path}' 进行推理...")
    
    # --- 1. 加载模型和分词器 ---
    print("正在加载模型和分词器...")
    try:
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            print(f"错误：模型路径不存在 -> {model_path}")
            return None
            
        tokenizer = AutoTokenizer.from_pretrained(model_path_obj)
        model = AutoModelForCausalLM.from_pretrained(
            model_path_obj,
            torch_dtype=torch.bfloat16, # 使用bfloat16以提高效率
            device_map="auto"           # 自动将模型分配到可用的GPU或CPU
        )
        model.eval()  # 将模型设置为评估模式
        print("模型加载成功。")
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return None

    # --- 2. 逐行推理并保存结果 ---
    input_path = Path(stance_jsonl_path)
    output_path = Path(output_jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) # 确保输出目录存在

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        lines = infile.readlines()
        
        for line in tqdm(lines, desc="正在运行微调模型推理"):
            data = json.loads(line)
            
            # 提取需要推理的 messages
            input_messages = data.get("messages", [])
            if not input_messages:
                continue
            
            # 调用辅助函数获取预测结果
            prediction = _get_prediction(model, tokenizer, input_messages)
            
            # 找到 "assistant" 角色并用预测结果填充其 "content"
            for message in input_messages:
                if message.get("role") == "assistant":
                    message["content"] = str(prediction) if prediction is not None else ""
            
            # 将更新后的数据写回文件
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"微调模型处理完成，输出文件: {output_path}")
    return str(output_path)
