import numpy as np
from typing import List

def calculate_deviation_index(user_stances: List[int], event_nature: int) -> float:
    """
    根据给定的公式计算事件严重程度 (Deviation Index, DI)。

    Args:
        user_stances (List[int]): 一个包含所有用户立场倾向的列表。
                                  每个元素的取值应为 1 (支持), 0 (中立), -1 (反对)。
        event_nature (int): 事件本身的性质。根据您的新要求，取值为 1 (正向), 0 (中立), 或 -1 (负向)。

    Returns:
        float: 计算出的 DI 分值。如果输入列表为空则返回 0.0。
    """
    # --- 输入校验 ---
    if not user_stances:
        return 0.0

    if event_nature not in [1, 0, -1]:
        raise ValueError("event_nature 必须是 1, 0, 或 -1")

    # --- 核心计算 ---
    # 将列表转换为 NumPy 数组以便进行高效的向量化计算
    stances_array = np.array(user_stances)

    # 计算每条内容的偏离度 |Pᵢ - P|
    deviations = np.abs(stances_array - event_nature)

    # 求所有偏离度的平均值，即 DI
    di_score = np.mean(deviations)

    return di_score

def classify_di_level(di_score: float) -> str:
    """
    根据 DI 分值将其转换为中文的偏离等级。

    Args:
        di_score (float): calculate_deviation_index 函数计算出的分值。

    Returns:
        str: 对应的中文等级描述。
    """
    # 注意：为了处理边界情况（如0.3, 0.6），我们采用左闭右开的区间判断，
    # 这是编程中的标准做法。例如 0.6 会被归类为“重度偏离”。
    if di_score < 0.3:
        return "轻度偏离"
    elif di_score < 0.6:
        return "中度偏离"
    elif di_score < 0.8:
        return "重度偏离"
    else:
        return "特重度偏离"

def get_event_severity(user_stances: List[int], event_nature: int) -> str:
    """
    一个完整的主函数，接收输入并直接返回最终的中文严重等级。

    Args:
        user_stances (List[int]): 用户立场倾向列表 [1, 0, -1, ...]。
        event_nature (int): 事件性质 1, 0, 或 -1。

    Returns:
        str: 中文的严重等级描述。
    """
    # 步骤1: 计算DI数值
    di_score = calculate_deviation_index(user_stances, event_nature)
    
    # 步骤2: 将DI数值转换为等级描述
    severity_level = classify_di_level(di_score)
    
    return severity_level

# ==================== 代码使用示例 ====================
# 类的对外接口get_event_severity(stances_1, event_p_1)
if __name__ == "__main__":

    # --- 示例1: 负向事件，舆论严重对立 ---
    print("--- 示例1: 负向事件，舆论严重对立 ---")
    event_p_1 = -1  # 例如：“校园霸凌事件”
    stances_1 = [-1, -1, 0, 1, 1, 1, 1, 1, 1, 1]  # 10条评论：2条反对, 1条中立, 7条支持
    
    # 直接调用主函数获取结果
    severity_1 = get_event_severity(stances_1, event_p_1)
    
    # (可选) 打印中间计算过程
    di_score_1 = calculate_deviation_index(stances_1, event_p_1)
    print(f"事件性质 P = {event_p_1}, 用户立场 Pᵢ = {stances_1}")
    print(f"计算出的 DI 值为: {di_score_1:.4f}")
    print(f"最终的事件严重程度为: 【{severity_1}】\n")
    # 预期输出：DI=1.5 -> 特重度偏离

    # --- 示例2: 正向事件，舆论基本符合 ---
    print("--- 示例2: 正向事件，舆论基本符合 ---")
    event_p_2 = 1   # 例如：“科技公司取得重大突破”
    stances_2 = [1, 1, 1, 1, 0, 0, -1, 1, 1, 1]  # 10条评论：8条支持, 2条中立, 1条反对
    
    severity_2 = get_event_severity(stances_2, event_p_2)
    di_score_2 = calculate_deviation_index(stances_2, event_p_2)
    print(f"事件性质 P = {event_p_2}, 用户立场 Pᵢ = {stances_2}")
    print(f"计算出的 DI 值为: {di_score_2:.4f}")
    print(f"最终的事件严重程度为: 【{severity_2}】\n")
    # 预期输出：DI=0.4 -> 中度偏离

    # --- 示例3: 中立事件，舆论两极分化 ---
    print("--- 示例3: 中立事件，舆论两极分化 ---")
    event_p_3 = 0   # 例如：“某公司发布中性财报”
    stances_3 = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1] # 10条评论：5条看好(支持), 5条看衰(反对)
    
    severity_3 = get_event_severity(stances_3, event_p_3)
    di_score_3 = calculate_deviation_index(stances_3, event_p_3)
    print(f"事件性质 P = {event_p_3}, 用户立场 Pᵢ = {stances_3}")
    print(f"计算出的 DI 值为: {di_score_3:.4f}")
    print(f"最终的事件严重程度为: 【{severity_3}】\n")
    # 预期输出：DI=1.0 -> 特重度偏离