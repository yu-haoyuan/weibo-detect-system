import os
import sys
import csv
from pathlib import Path
import time

# 从components包中导入所有需要的处理函数
# (因为我们更新了 __init__.py, 所以可以这样简洁地导入)
from components import (
    convert_csv_to_stance_jsonl,
    convert_csv_to_sensitivity_jsonl,
    process_with_finetuned_model,
    process_with_api_model,
    calculate_event_severity,
    calculate_event_impact
)

# ==============================================================================
# 全局配置 (Global Configuration)
# ==============================================================================
# --- 微调模型配置 ---
FINETUNED_MODEL_PATH = "/mnt/sdb/tangchengxiang/xxq/weibo-detect-system/qwen2.5-0.5b-ft-stand_detect"

# --- API模型配置 ---
# VVVVVV 在这里填入您的API信息 VVVVVV
API_KEY = "sk-b386510df5614cb1bdde83e71f65e7b0"
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_MODEL_NAME = "qwen-plus"
# 这个文件包含了用于 few-shot 学习的示例
FEW_SHOT_EXAMPLES_PATH = "/mnt/sdb/tangchengxiang/xxq/weibo-detect-system/old_meta_data/train.csv"
# ^^^^^^ 在这里填入您的API信息 ^^^^^^


# ==============================================================================
# 主执行流程 (Main Pipeline)
# ==============================================================================
def run_pipeline(csv_input_file_path: str, output_dir_path: str):
    """
    主函数，按顺序调用所有组件，执行完整的数据处理流水线。
    现在它接收输入和输出路径作为参数，并返回一个包含结果的字典。
    """
    print("===== 数据处理流水线开始 =====")
    
    CSV_INPUT_FILE = Path(csv_input_file_path)
    OUTPUT_DIR = Path(output_dir_path)
    
    # 检查并创建输出文件夹 (Flask部分已创建，这里确保万无一失)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"所有输出将保存在: {OUTPUT_DIR.resolve()}")

    # --- 定义中间及输出文件路径 ---
    STANCE_JSONL_FILE = OUTPUT_DIR / 'stance_tendency.jsonl'
    SENSITIVITY_JSONL_FILE = OUTPUT_DIR / 'event_sensitivity.jsonl'
    CONTENT_FROM_FINETUNE_FILE = OUTPUT_DIR / 'content_from_finetune.jsonl'
    CONTENT_FROM_API_FILE = OUTPUT_DIR / 'content_from_api.jsonl'
    SEVERITY_OUTPUT_FILE = OUTPUT_DIR / 'final_event_severity.txt'
    IMPACT_OUTPUT_FILE = OUTPUT_DIR / 'final_event_impact.txt'
    FINAL_SUMMARY_CSV = OUTPUT_DIR / f'summary_report_{CSV_INPUT_FILE.stem}.csv'

    # --- 阶段 1: 预处理 ---
    print("\n--- 阶段 1: 正在进行数据预处理 ---")
    convert_csv_to_stance_jsonl(CSV_INPUT_FILE, STANCE_JSONL_FILE)
    convert_csv_to_sensitivity_jsonl(CSV_INPUT_FILE, SENSITIVITY_JSONL_FILE)

    # --- 阶段 2: 模型推理 ---
    print("\n--- 阶段 2: 正在运行模型推理 ---")
    
    process_with_finetuned_model(
        stance_jsonl_path=STANCE_JSONL_FILE, 
        output_jsonl_path=CONTENT_FROM_FINETUNE_FILE, 
        model_path=FINETUNED_MODEL_PATH
    )
    
    process_with_api_model(
        sensitivity_jsonl_path=SENSITIVITY_JSONL_FILE, 
        output_jsonl_path=CONTENT_FROM_API_FILE, 
        api_key=API_KEY, 
        base_url=API_BASE_URL, 
        model_name=API_MODEL_NAME,
        few_shot_examples_path=FEW_SHOT_EXAMPLES_PATH
    )

    # --- 阶段 3: 分析与整合 ---
    print("\n--- 阶段 3: 正在进行分析与整合 ---")
    di_scores_timed, final_di_score, severity_level, topic_stance, sensitivity = calculate_event_severity(
        CONTENT_FROM_FINETUNE_FILE,
        CONTENT_FROM_API_FILE, 
        str(SEVERITY_OUTPUT_FILE)
    )
    
    heat_data, urgency_level, impact_level = calculate_event_impact(CSV_INPUT_FILE, IMPACT_OUTPUT_FILE)

    # --- 阶段 4: 结果汇总 ---
    print("\n--- 阶段 4: 正在汇总结果 ---")
    
    results_dict = {
        '文件名': CSV_INPUT_FILE.name,
        '1小时热度': heat_data.get('1小时热度', 'N/A'),
        '2小时热度': heat_data.get('2小时热度', 'N/A'),
        '3小时热度': heat_data.get('3小时热度', 'N/A'),
        '4小时热度': heat_data.get('4小时热度', 'N/A'),
        '6小时热度': heat_data.get('6小时热度', 'N/A'),
        '9小时热度': heat_data.get('9小时热度', 'N/A'),
        '总热度': heat_data.get('总热度', 'N/A'),
        '严重系数1小时': di_scores_timed.get('严重系数1小时', 'N/A'),
        '严重系数2小时': di_scores_timed.get('严重系数2小时', 'N/A'),
        '严重系数3小时': di_scores_timed.get('严重系数3小时', 'N/A'),
        '严重程度': severity_level,
        '话题倾向': topic_stance,
        '敏感程度': sensitivity,
        '紧急程度': urgency_level,
        '影响范围': impact_level
    }

    # 写入本次处理的独立CSV报告
    header = list(results_dict.keys())
    try:
        with open(FINAL_SUMMARY_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerow(results_dict)
        print(f"单个任务报告已保存到: {FINAL_SUMMARY_CSV}")
        # 在返回的字典中也包含这个路径，方便前端生成下载链接
        results_dict['final_summary_report_path'] = str(FINAL_SUMMARY_CSV)
    except IOError as e:
        print(f"\n[错误] 写入CSV文件失败: {e}")
        results_dict['final_summary_report_path'] = None

    print("\n===== 数据处理流水线全部完成 =====")
    
    # 返回结果字典
    return results_dict
