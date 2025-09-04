import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

# --- 内部核心算法 (从之前版本继承) ---

def _calculate_deviation_index(user_stances: List[int], event_nature: int) -> float:
    """根据公式计算事件严重程度 (Deviation Index, DI)。"""
    if not user_stances:
        return 0.0
    stances_array = np.array(user_stances)
    deviations = np.abs(stances_array - event_nature)
    return np.mean(deviations)

def _classify_di_level(di_score: float) -> str:
    """根据 DI 分值将其转换为中文的偏离等级。"""
    if di_score < 0.3:
        return "轻度偏离"
    elif di_score < 0.6:
        return "中度偏离"
    elif di_score < 0.8:
        return "重度偏离"
    else:
        return "特重度偏离"

# --- 主要组件函数 (已重构) ---

def calculate_event_severity(
    finetune_results_path: str, 
    api_results_path: str, 
    output_report_path: str
) -> Tuple[Dict, float, str, str, str]:
    """
    整合模型输出，计算最终的事件严重程度、分时段严重系数和总体话题倾向。
    """
    print("开始分析事件严重程度...")
    
    # --- 1. 读取API模型结果，确定事件性质和敏感度 ---
    try:
        with open(api_results_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line: raise ValueError("API结果文件为空。")
            api_data = json.loads(first_line)
            assistant_msg = next((m for m in api_data['messages'] if m['role'] == 'assistant'), {})
            sensitivity = assistant_msg.get('content', '-1')
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"读取或解析API结果文件时出错: {e}")
        return {}, 0.0, "未知", "未知", "未知"

    # 根据敏感度映射事件性质 P (用于DI计算)
    if sensitivity in ['1', '2', '3']:
        event_nature = -1  # 负向事件
    else: # sensitivity '0' or invalid
        event_nature = 0   # 中立事件

    # --- 2. 读取微调模型结果（包含立场和时间戳） ---
    records = []
    try:
        with open(finetune_results_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 重要：假设 stance_csv_to_jsonl 组件已将时间戳加入metadata
                timestamp = data.get("metadata", {}).get("timestamp")
                assistant_msg = next((m for m in data['messages'] if m['role'] == 'assistant'), None)
                if assistant_msg and assistant_msg['content'] in ['-1', '0', '1']:
                    records.append({
                        "timestamp": timestamp,
                        "stance": int(assistant_msg['content'])
                    })
    except (FileNotFoundError, json.JSONDecodeError):
        print("读取微调结果文件失败或文件格式错误。")
        return {}, 0.0, "未知", "未知", sensitivity

    if not records or all(r['timestamp'] is None for r in records):
        print("警告：微调结果中无有效数据或时间戳。无法计算分时段严重系数。")
        return {}, 0.0, "未知", "未知", sensitivity

    # --- 3. 数据处理与计算 ---
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    
    if df.empty:
        return {}, 0.0, "未知", "未知", sensitivity

    all_stances = df['stance'].tolist()
    start_time = df['timestamp'].iloc[0]

    # 计算最终的DI和严重等级
    final_di_score = _calculate_deviation_index(all_stances, event_nature)
    final_severity_level = _classify_di_level(final_di_score)
    
    # 计算分时段的严重系数
    di_scores_timed = {}
    for h in [1, 2, 3]:
        target_time = start_time + pd.Timedelta(hours=h)
        stances_in_period = df[df['timestamp'] <= target_time]['stance'].tolist()
        di_timed = _calculate_deviation_index(stances_in_period, event_nature)
        di_scores_timed[f'严重系数{h}小时'] = f"{di_timed:.4f}"

    # 计算总体话题倾向
    avg_stance = np.mean(all_stances)
    if avg_stance > 0.33: topic_stance =1
    elif avg_stance < -0.33: topic_stance =-1
    else: topic_stance =0

    # --- 4. 写入文本报告并返回结果 ---
    report_content = (
        f"事件严重程度分析报告\n"
        f"=========================\n"
        f"API判定敏感度: {sensitivity}\n"
        f"映射事件性质 (P): {event_nature}\n"
        f"总体话题倾向: {topic_stance}\n"
        f"-------------------------\n"
        f"最终严重系数 (DI): {final_di_score:.4f}\n"
        f"最终严重等级: 【{final_severity_level}】\n"
        f"-------------------------\n"
        f"分时段严重系数:\n"
        f"  - 1小时内: {di_scores_timed['严重系数1小时']}\n"
        f"  - 2小时内: {di_scores_timed['严重系数2小时']}\n"
        f"  - 3小时内: {di_scores_timed['严重系数3小时']}\n"
    )
    Path(output_report_path).write_text(report_content, encoding='utf-8')
    print("事件严重程度分析完成。")

    return di_scores_timed, final_di_score, final_severity_level, topic_stance, sensitivity

