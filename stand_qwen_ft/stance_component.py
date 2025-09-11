# stance_component.py

import json
import re
from tqdm import tqdm
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

class StanceDetector:
    """
    一个用于立场判断的推理组件。
    它在初始化时加载模型，并提供接口进行预测。
    """
    def __init__(self, model_path: str):
        """
        加载并初始化模型和tokenizer。
        这是一个耗时操作，只应在程序启动时执行一次。
        """
        print(f"正在从 '{model_path}' 加载模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" # 自动将模型分配到可用GPU
        )
        self.model.eval()
        print("模型加载完成，组件已就绪。")

    def predict(self, messages: List[Dict[str, Any]]) -> str:
        """
        对单条对话消息进行立场判断。
        """
        # 1. 准备输入
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # 2. 模型推理
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=10,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 3. 解码和解析结果
        decoded_output = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        match = re.search(r"(-?\d)", decoded_output)
        
        if match:
            return match.group(1)
        else:
            # 如果模型没有返回期望的数字，可以返回一个默认值或错误标识
            return "prediction_error"

    def predict_from_file(self, input_path: str, output_path: str):
        """
        对一个JSONL文件进行批量推理，并将结果写入新文件。
        """
        print(f"开始批量处理文件: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            lines = infile.readlines()
            
            for line in tqdm(lines, desc="批量推理中"):
                data = json.loads(line)
                
                # 获取用于推理的消息 (不包含空的assistant部分)
                input_messages = [msg for msg in data.get("messages", []) if msg.get("role") != "assistant"]
                
                # 调用单条预测方法
                prediction = self.predict(input_messages)
                
                # 更新原始数据中的assistant content
                for message in data.get("messages", []):
                    if message.get("role") == "assistant":
                        message["content"] = str(prediction)
                
                # 写入更新后的结果
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
        print(f"批量推理完成，结果已保存至: {output_path}")