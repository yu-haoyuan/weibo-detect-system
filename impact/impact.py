import pandas as pd
import numpy as np
import argparse
# import joblib  # 如果您使用 scikit-learn/xgboost, 用这个库来加载模型

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    加载并预处理CSV数据。
    1. 读取CSV。
    2. 将 '发布时间' 转换为datetime对象。
    3. 按时间升序排序。
    """
    print("1. 正在加载和预处理数据...")
    try:
        # 读取CSV，并指定数据类型以避免警告
        df = pd.read_csv(
            file_path,
            dtype={'转发数': int, '评论数': int, '点赞数': int}
        )
        
        # 【速度优化】为 to_datetime 指定确切的日期格式，避免自动解析，提升速度
        df['发布时间'] = pd.to_datetime(df['发布时间'], format='%Y-%m-%d %H:%M', errors='coerce')
        
        # 移除无法解析时间的行并排序
        df.dropna(subset=['发布时间'], inplace=True)
        df.sort_values(by='发布时间', inplace=True, ignore_index=True)
        
        print("   - 数据加载和排序完成。")
        return df
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except KeyError as e:
        print(f"错误：CSV文件中缺少必需的列: {e}")
        return None

def calculate_heat_over_time(df: pd.DataFrame) -> dict:
    """
    计算在指定时间点的累计热度。
    """
    print("2. 正在计算各时间点热度...")
    if df.empty:
        return {}
        
    # 根据定义计算每一条微博的独立热度
    # 【已修正】热度 = 发帖量(1) + 点赞量 + 转发量 + 评论数
    df['individual_heat'] = 1 + df['点赞数'] + df['转发数'] + df['评论数']
    
    # 计算累计热度
    df['cumulative_heat'] = df['individual_heat'].cumsum()
    
    # 获取事件的起始时间
    start_time = df['发布时间'].iloc[0]
    
    # 【速度优化】使用 searchsorted 替换循环中的布尔索引，以提高在大数据集上的计算效率
    heat_results = {}
    last_heat = 0
    intervals_hours = [1, 2, 3, 4, 6, 9]

    for h in intervals_hours:
        target_time = start_time + pd.Timedelta(hours=h)
        
        # searchsorted 是一种高效的查找算法，用于在有序数组中找到元素的插入点
        idx = df['发布时间'].searchsorted(target_time, side='right')
        
        # 如果找到了至少一个符合条件的帖子 (idx > 0)
        if idx > 0:
            # 更新当前热度为该时间点最后一篇帖子的累计热度
            last_heat = df['cumulative_heat'].iloc[idx - 1]
        
        # 记录当前小时的热度。如果该时间段内没有新帖子, last_heat 会保持之前的值
        heat_results[f'{h}小时热度'] = last_heat
            
    # 总热度是最后一篇帖子的累计热度
    heat_results['总热度'] = df['cumulative_heat'].iloc[-1]
    
    print("   - 热度计算完成。")
    return heat_results

def predict_urgency(heat_3h: int, heat_6h: int, heat_9h: int) -> str:
    """
    【模型优化】根据3h, 6h, 9h的热度，通过二次函数拟合预测第24小时的紧急程度。
    """
    print("3. 正在预测事件紧急程度...")
    
    # --- 基于二次函数拟合的预测逻辑 ---
    # 我们有三个点: (3, heat_3h), (6, heat_6h), (9, heat_9h)
    # 拟合曲线 H(t) = at^2 + bt + c
    # 我们要求解的是在 t=24 时的瞬时增长率 H'(24)，其中 H'(t) = 2at + b
    
    try:
        # 计算系数 a (代表增长的加速度)
        a = (heat_9h - 2 * heat_6h + heat_3h) / 18.0
        
        # 计算系数 b
        b = ((heat_6h - heat_3h) / 3.0) - (9 * a)
        
        # 预测第24小时的瞬时增长率: H'(24) = 2*a*24 + b
        predicted_growth_rate = 48 * a + b
        
        # 增长率不应为负数，如果趋势向下，则认为增长率为0
        predicted_growth_rate = max(0, predicted_growth_rate)

    except ZeroDivisionError:
        # 避免潜在的除零错误
        predicted_growth_rate = 0

    print(f"   - (模型预测) 预测的24小时单位时间增长速度: {predicted_growth_rate:.2f}")

    # 根据您的阈值进行分类
    if predicted_growth_rate <= 1000:
        urgency = "不紧急"
    elif predicted_growth_rate <= 5000:
        urgency = "紧急"
    elif predicted_growth_rate <= 10000:
        urgency = "较紧急"
    else:
        urgency = "非常紧急"
        
    print("   - 预测完成。")
    return urgency

def predict_final_impact(heat_3h: int, heat_6h: int, heat_9h: int) -> dict:
    """
    【新增】根据3h, 6h, 9h的热度，预测最终总热度并划分事件影响范围。
    """
    print("4. 正在预测最终事件影响范围...")

    try:
        # --- 复用二次函数拟合模型 H(t) = at^2 + bt + c ---
        # 计算系数 a 和 b (与 predict_urgency 中逻辑相同)
        a = (heat_9h - 2 * heat_6h + heat_3h) / 18.0
        b = ((heat_6h - heat_3h) / 3.0) - (9 * a)
        
        # 计算系数 c
        # c = H(3) - 9a - 3b
        c = heat_3h - 9 * a - 3 * b
        
        # 预测第24小时的累计热度 H(24) = a*24^2 + b*24 + c
        predicted_heat_24h = a * (24**2) + b * 24 + c
        
        # 预测的最终热度不应小于已知的最新热度
        predicted_final_heat = max(predicted_heat_24h, heat_9h)
        predicted_final_heat = int(predicted_final_heat)

    except ZeroDivisionError:
        predicted_final_heat = heat_9h

    # 根据总热度预测值划分影响力范围
    if predicted_final_heat <= 10000:
        scope = "小"
    elif predicted_final_heat <= 100000:
        scope = "中"
    elif predicted_final_heat <= 1000000:
        scope = "大"
    else:
        scope = "超大"

    print(f"   - (模型预测) 预测的最终总热度: {predicted_final_heat}")
    print(f"   - (模型预测) 预测的影响范围: 【{scope}】")

    return {"predicted_total_heat": predicted_final_heat, "impact_scope": scope}

def analyze_event(file_path: str) -> dict:
    """
    执行完整的事件分析流程，并返回包含所有结果的字典。
    """
    df = load_and_preprocess_data(file_path)
    
    if df is None or df.empty:
        if df is not None and df.empty:
             print("\n错误：CSV文件处理后没有数据。请检查'发布时间'列的格式是否正确，或文件内容是否为空。")
        return None

    heat_results = calculate_heat_over_time(df)
    
    # 从热度结果中提取预测所需的输入
    heat_3h = heat_results.get('3小时热度', 0)
    heat_6h = heat_results.get('6小时热度', 0)
    heat_9h = heat_results.get('9小时热度', 0)
    
    urgency_prediction = predict_urgency(heat_3h, heat_6h, heat_9h)
    impact_prediction = predict_final_impact(heat_3h, heat_6h, heat_9h)
    
    # 将所有预测结果也添加到结果字典中
    heat_results['预测的紧急程度'] = urgency_prediction
    heat_results['预测的最终总热度'] = impact_prediction['predicted_total_heat']
    heat_results['预测的影响范围'] = impact_prediction['impact_scope']
    
    return heat_results

def main(file_path: str):
    """
    主函数，调用分析流程并打印最终报告。
    """
    analysis_results = analyze_event(file_path)
    print(analysis_results)
    if analysis_results:
        # --- 打印最终结果 ---
        print("\n" + "="*40)
        print("          事件热度与紧急度分析报告")
        print("="*40)
        
        output_order = [
            '1小时热度', '2小时热度', '3小时热度',
            '4小时热度', '6小时热度', '9小时热度', '总热度'
        ]
        
        for key in output_order:
            value = analysis_results.get(key, '未计算')
            print(f"{key:<12}: {value}")
            
        print("-" * 40)
        urgency = analysis_results.get('预测的紧急程度', '未知')
        predicted_heat = analysis_results.get('预测的最终总热度', 0)
        scope = analysis_results.get('预测的影响范围', '未知')

        print(f"{'预测的紧急程度':<12}: 【{urgency}】")
        print(f"{'预测的最终总热度':<12}: {predicted_heat}")
        print(f"{'预测的影响范围':<12}: 【{scope}】")
        print("="*40)

if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='分析微博事件CSV文件，计算热度并预测紧急程度。')
    parser.add_argument('file_path', type=str, help='输入的CSV文件路径')
    
    args = parser.parse_args()
    
    main(args.file_path)

