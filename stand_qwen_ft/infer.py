import json
import re
from tqdm import tqdm
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_prediction(model, tokenizer, messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=10,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded_output = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    match = re.search(r"(-?\d)", decoded_output)
    
    if match:
        return match.group(1)
    else:
        return None

def process_inference_data(input_path, output_path, model, tokenizer):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        lines = infile.readlines()
        
        for line in tqdm(lines, desc="Running Inference"):
            data = json.loads(line)
            
            input_messages = data.get("messages", [])
            
            prediction = get_prediction(model, tokenizer, input_messages)
            
            for message in input_messages:
                if message.get("role") == "assistant" and "content" in message:
                    message["content"] = str(prediction) if prediction else ""
            
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

def main():
    #change this
    model_path = Path("/mnt/sdb/tangchengxiang/xxq/weibo-detect-system/qwen2.5-0.5b-ft-stand_detect")
    input_file = Path("/mnt/sdb/tangchengxiang/xxq/weibo-detect-system/stand_qwen_ft/data/test_infer.jsonl")
    output_file = Path("/mnt/sdb/tangchengxiang/xxq/weibo-detect-system/stand_qwen_ft/data/test_pred.jsonl")
    #change end

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print("Model loaded.")

    process_inference_data(input_file, output_file, model, tokenizer)
    print("Inference completed and results saved.")

if __name__ == "__main__":
    main()