# 微博事件分析流水线 - 目录结构与组件说明

## 📂 项目目录结构
```bash
.
├── components/                 # 核心功能组件目录
│   ├── __init__.py             # 组件函数导出声明
│   ├── stance_csv_to_jsonl.py  # 立场检测数据格式转换（CSV→JSONL）
│   ├── sensitivity_csv_to_jsonl.py  # 敏感度分析数据格式转换（CSV→JSONL）
│   ├── stance.py               # 本地微调模型推理（立场检测）
│   ├── sensitivity.py          # 云端API模型推理（敏感度分析）
│   ├── calculate_severity.py   # 事件严重程度计算与分析
│   └── calculate_impact.py     # 事件影响范围与紧急度计算
├── output/                     # 输出文件目录（自动创建）
│   ├── stance_tendency.jsonl   # 立场检测模型输入数据
│   ├── event_sensitivity.jsonl # 敏感度分析模型输入数据
│   ├── content_from_finetune.jsonl  # 本地模型推理结果
│   ├── content_from_api.jsonl      # 云端API模型推理结果
│   ├── final_event_severity.txt    # 事件严重程度报告
│   ├── final_event_impact.txt      # 事件影响范围报告
│   └── final_summary_report.csv    # 所有指标汇总CSV
├── main.py                     # 主执行脚本（调度整个流水线）
└── README.md                   # 项目说明文档
```


## 🔍 核心组件功能与实现说明

### 1. 数据格式转换组件（CSV→JSONL）
#### `stance_csv_to_jsonl.py`
- **功能**：将输入的微博数据CSV文件转换为本地微调模型可识别的JSONL格式（用于立场检测）。
- **实现方式**：
  1. 读取CSV中的“话题”“微博正文”“发布时间”字段；
  2. 按固定模板生成模型输入（包含系统提示、用户提问），系统提示定义立场分类规则（-1反对/0中立/1支持）；
  3. 将每条微博数据转换为包含`messages`（模型输入）和`metadata`（时间戳）的JSONL条目，写入输出文件。


#### `sensitivity_csv_to_jsonl.py`
- **功能**：将输入的CSV文件转换为云端API模型可识别的JSONL格式（用于敏感度分析）。
- **实现方式**：
  1. 提取CSV文件的文件名作为“话题名称”（整个文件对应一个事件话题）；
  2. 按固定模板生成模型输入（包含系统提示、用户提问），系统提示定义敏感度分类规则（0-3级）；
  3. 为整个CSV文件生成一个JSONL条目（包含`messages`），写入输出文件。


### 2. 模型推理组件
#### `stance.py`
- **功能**：使用本地微调模型对立场检测JSONL数据进行推理，输出每条微博的立场标签（-1/0/1）。
- **实现方式**：
  1. 加载本地微调模型（基于`transformers`库）和分词器；
  2. 逐行读取`stance_tendency.jsonl`中的模型输入，调用模型生成结果；
  3. 通过正则表达式提取模型输出中的数字（-1/0/1），填充到`assistant`角色的`content`字段；
  4. 将结果写入`content_from_finetune.jsonl`。


#### `sensitivity.py`
- **功能**：调用云端API模型对敏感度分析JSONL数据进行推理，输出事件的敏感度等级（0-3）。
- **实现方式**：
  1. 初始化云端API客户端（如阿里云DashScope），加载few-shot示例文件（提升模型准确性）；
  2. 读取`sensitivity_jsonl.py`生成的JSONL，提取话题名称；
  3. 构建包含系统提示、few-shot示例和当前话题的请求，调用API模型；
  4. 提取模型返回的敏感度等级（0-3），填充到`assistant`角色的`content`字段；
  5. 将结果写入`content_from_api.jsonl`。


### 3. 分析计算组件
#### `calculate_severity.py`
- **功能**：整合模型推理结果，计算事件严重程度、分时段严重系数和总体话题倾向。
- **实现方式**：
  1. 从`content_from_api.jsonl`获取敏感度等级，映射事件性质（负向事件/中立事件）；
  2. 从`content_from_finetune.jsonl`提取每条微博的立场标签和时间戳；
  3. 计算“偏离指数（DI）”：衡量用户立场与事件性质的偏离程度，转换为严重等级（轻度/中度/重度/特重度）；
  4. 按1/2/3小时分段计算严重系数，统计总体话题倾向（支持/反对/中立）；
  5. 生成文本报告`final_event_severity.txt`。


#### `calculate_impact.py`
- **功能**：基于原始CSV数据，计算事件热度、紧急程度和影响范围。
- **实现方式**：
  1. 读取CSV中的“发布时间”“转发数”“评论数”“点赞数”，计算单条微博热度（1+点赞+转发+评论）；
  2. 按1/2/3/4/6/9小时分段统计累计热度，预测24小时总热度；
  3. 基于热度增长趋势预测紧急程度（不紧急/紧急/较紧急/非常紧急）；
  4. 根据预测总热度划分影响范围（小/中/大/超大）；
  5. 生成文本报告`final_event_impact.txt`。


### 4. 主调度脚本 `main.py`
- **功能**：串联所有组件，执行端到端流水线。
- **执行流程**：
  1. 读取全局配置（输入文件路径、模型路径、API密钥等）；
  2. 调用转换组件生成模型输入JSONL；
  3. 调用本地模型和API模型执行推理；
  4. 调用分析组件计算严重程度和影响范围；
  5. 将所有指标（热度、严重系数、敏感程度等）汇总到`final_summary_report.csv`。


### 5. 输出目录 `output/`
- 存储所有中间结果（模型输入/输出JSONL）和最终报告（文本+CSV），便于追溯数据流转和结果分析。