# 微博检测系统V2.0

## 项目框架与说明

### 项目概述
微博检测系统V2.0是一个针对微博内容进行多维度分析的工具，能够自动识别微博内容的立场倾向、敏感程度，评估事件的严重程度和影响范围，为用户提供全面的舆情分析报告。

### 系统架构
```
weibo-detect-system-xiangmuV2.0/
├── app.py                  # Flask Web应用主程序
├── processing_pipeline.py  # 数据处理流水线主函数
├── components/             # 核心处理组件
│   ├── __init__.py
│   ├── stance_csv_to_jsonl.py  # 立场分析数据格式转换
│   ├── sensitivity_csv_to_jsonl.py  # 敏感度分析数据格式转换
│   ├── stance.py           # 立场倾向分析（使用微调模型）
│   ├── sensitivity.py      # 敏感程度分析（使用API模型）
│   ├── calculate_severity.py  # 事件严重程度计算
│   └── calculate_impact.py    # 事件影响范围评估
├── templates/              # 前端模板
│   └── index.html          # 主页面
├── stand_qwen_ft/          # Qwen模型微调相关
│   ├── README.md           # 微调说明文档
│   ├── data/               # 数据处理脚本
│   │   ├── split_ft.py     # 数据集拆分
│   │   └── make_test_stan_jsonl.py  # 生成测试数据
│   └── infer.py            # 推理脚本
├── uploads/                # 上传文件存储目录
├── output/                 # 分析结果存储目录
├── requirements.txt        # 项目依赖
└── README.md               # 项目说明文档
```

### 核心功能模块
1. **数据预处理模块**：将CSV格式的微博数据转换为模型可处理的JSONL格式
2. **立场倾向分析模块**：使用微调的Qwen2.5模型判断微博内容的立场（-1:反对, 0:中立, 1:支持）
3. **敏感程度分析模块**：通过API调用判断内容敏感级别（0-3级）
4. **事件严重程度评估**：综合多维度指标评估事件严重程度
5. **事件影响范围分析**：分析事件的热度变化和影响范围
6. **Web交互界面**：提供文件上传、结果展示和报告下载功能

## 使用方法

### 环境准备

1. 克隆仓库
```bash
git clone <仓库地址>
cd weibo-detect-system-xiangmuV2.0
```

2. 创建并激活虚拟环境
```bash
使用conda 或 miniconda
```

3. 安装依赖（有待完善）
```bash
pip install -r requirements.txt
```

4. 准备模型
   - 按照`stand_qwen_ft/README.md`中的说明安装或者自行微调模型

5. 配置API信息
   - 编辑`processing_pipeline.py`文件，填入API_KEY、API_BASE_URL等信息

### 启动应用

```bash
python app.py
```

应用将在本地5001端口启动，访问`http://localhost:5001`即可使用系统。

### 使用流程

1. 在Web界面点击"上传文件"按钮，选择包含微博数据的CSV文件
2. 点击"开始分析"按钮，系统将自动进行处理
3. 等待分析完成后，查看结果页面展示的分析报告
4. 可通过"下载完整报告"按钮获取CSV格式的详细报告
5. 点击"继续分析下一个文件"可上传新的文件进行分析

## 模型微调指南

如需自行微调立场分析模型，请参考`stand_qwen_ft/README.md`中的详细步骤，主要流程包括：

1. 配置LLaMA-Factory环境
2. 下载Qwen2.5-0.5B-Instruct模型
3. 准备并处理训练数据
4. 配置并执行微调训练
5. 导出并使用微调后的模型

## requirements.txt（不准确，gpt生成，后续若有人安装，可以完善上传）

```
flask==2.3.3
pandas==2.1.4
numpy==1.26.3
torch==2.1.2
transformers==4.36.2
tqdm==4.66.1
openai==1.3.5
python-dotenv==1.0.0
pathlib==1.0.1
werkzeug==2.3.7
```

## 注意事项

1. 系统仅支持CSV格式的输入文件，且需要包含"话题"和"微博正文"列
2. 进行敏感度分析需要有效的API密钥
3. 模型推理过程可能需要较长时间，具体取决于输入数据量和硬件性能
4. 首次运行时会下载相关模型文件，可能需要较长时间和稳定的网络连接
5. 建议在具有GPU支持的环境下运行，以提高模型推理速度

## 联系方式

如有任何问题或建议，请联系项目维护人员。