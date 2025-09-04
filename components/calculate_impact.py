import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# --- 内部核心算法 (来自您的脚本) ---

def _load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    加载并预处理CSV数据。这是一个内部函数。
    """
    print("正在加载和预处理影响数据...")
    try:
        df = pd.read_csv(
            file_path,
            dtype={'转发数': int, '评论数': int, '点赞数': int}
        )
        df['发布时间'] = pd.to_datetime(df['发布时间'], format='%Y-%m-%d %H:%M', errors='coerce')
        df.dropna(subset=['发布时间'], inplace=True)
        df.sort_values(by='发布时间', inplace=True, ignore_index=True)
        return df
    except FileNotFoundError:
        print(f"错误：找不到影响分析所需的文件 {file_path}")
        return None
    except KeyError as e:
        print(f"错误：影响分析所需的CSV文件中缺少必需的列: {e}")
        return None

def _calculate_heat_over_time(df: pd.DataFrame) -> dict:
    """
    计算在指定时间点的累计热度。这是一个内部函数。
    """
    if df.empty:
        return {}
    df['individual_heat'] = 1 + df['点赞数'] + df['转发数'] + df['评论数']
    df['cumulative_heat'] = df['individual_heat'].cumsum()
    start_time = df['发布时间'].iloc[0]
    
    heat_results = {}
    last_heat = 0
    intervals_hours = [1, 2, 3, 4, 6, 9]

    for h in intervals_hours:
        target_time = start_time + pd.Timedelta(hours=h)
        idx = df['发布时间'].searchsorted(target_time, side='right')
        if idx > 0:
            last_heat = df['cumulative_heat'].iloc[idx - 1]
        heat_results[f'{h}小时热度'] = last_heat
            
    heat_results['总热度'] = df['cumulative_heat'].iloc[-1]
    return heat_results

def _predict_urgency(heat_3h: int, heat_6h: int, heat_9h: int) -> str:
    """
    根据3h, 6h, 9h的热度，预测紧急程度。这是一个内部函数。
    """
    try:
        a = (heat_9h - 2 * heat_6h + heat_3h) / 18.0
        b = ((heat_6h - heat_3h) / 3.0) - (9 * a)
        predicted_growth_rate = max(0, 48 * a + b)
    except ZeroDivisionError:
        predicted_growth_rate = 0

    if predicted_growth_rate <= 1000: return "不紧急"
    if predicted_growth_rate <= 5000: return "紧急"
    if predicted_growth_rate <= 10000: return "较紧急"
    return "非常紧急"

def _predict_final_impact(heat_3h: int, heat_6h: int, heat_9h: int) -> Tuple[int, str]:
    """
    根据3h, 6h, 9h的热度，预测最终总热度并划分事件影响范围。这是一个内部函数。
    """
    try:
        a = (heat_9h - 2 * heat_6h + heat_3h) / 18.0
        b = ((heat_6h - heat_3h) / 3.0) - (9 * a)
        c = heat_3h - 9 * a - 3 * b
        predicted_heat_24h = a * (24**2) + b * 24 + c
        predicted_final_heat = int(max(predicted_heat_24h, heat_9h))
    except ZeroDivisionError:
        predicted_final_heat = heat_9h

    if predicted_final_heat <= 10000: scope = "小"
    elif predicted_final_heat <= 100000: scope = "中"
    elif predicted_final_heat <= 1000000: scope = "大"
    else: scope = "超大"
    
    return predicted_final_heat, scope

# --- 主要组件函数 (流水线接口) ---

def calculate_event_impact(
    csv_path: str, 
    output_path: str
) -> Tuple[Dict, str, str]:
    """
    执行完整的事件影响分析流程，将报告写入文件，并返回核心结果。
    
    Args:
        csv_path (str): 原始输入的.csv文件路径。
        output_path (str): 最终分析报告的输出路径。

    Returns:
        Tuple[Dict, str, str]: 一个包含三项内容的元组：
            - heat_results (Dict): 包含各时间点热度的字典。
            - urgency (str): 预测的紧急程度。
            - impact_scope (str): 预测的影响范围。
    """
    print("开始分析事件影响程度...")
    df = _load_and_preprocess_data(csv_path)
    
    # 检查数据是否有效
    if df is None or df.empty:
        if df is not None and df.empty:
            print("错误：CSV文件处理后没有数据，无法进行影响分析。")
        # 返回默认的空值
        return {}, "未知", "未知"

    heat_results = _calculate_heat_over_time(df)
    
    heat_3h = heat_results.get('3小时热度', 0)
    heat_6h = heat_results.get('6小时热度', 0)
    heat_9h = heat_results.get('9小时热度', 0)
    
    urgency = _predict_urgency(heat_3h, heat_6h, heat_9h)
    predicted_heat, impact_scope = _predict_final_impact(heat_3h, heat_6h, heat_9h)

    # --- 将结果写入文件 ---
    try:
        output_file = Path(output_path)
        with output_file.open('w', encoding='utf-8') as f:
            f.write("="*40 + "\n")
            f.write("          事件影响程度与紧急度分析报告\n")
            f.write("="*40 + "\n")
            
            for key, value in heat_results.items():
                f.write(f"{key:<12}: {value}\n")
            
            f.write("-" * 40 + "\n")
            f.write(f"{'预测的紧急程度':<12}: 【{urgency}】\n")
            f.write(f"{'预测的最终总热度':<12}: {predicted_heat}\n")
            f.write(f"{'预测的影响范围':<12}: 【{impact_scope}】\n")
            f.write("="*40 + "\n")
        print(f"事件影响程度分析完成，结果已保存在: {output_path}")

    except Exception as e:
        print(f"错误：无法将影响程度报告写入到输出文件 {output_path}。错误: {e}")

    return heat_results, urgency, impact_scope
