# weibo-detect-system

[数据和项目源地址](https://tianchi.aliyun.com/competition/entrance/532363/information)

**对于所有脚本，都需要自行修改路径，如果能帮助整理脚本和路径格式非常感谢🙏**

todolist
---
事件分析任务清单
- [x] 立场倾向判断 (微调模型)

- [x] 事件敏感程度 (调用大模型)

[ ] 事件严重程度 (根据前面两个结果计算)

[ ] 事件紧急程度 (待定)

[ ] 事件影响范围 (待定)

框架

[x] 模型微调 (qwen 0.n b)

[ ] 高速推理 (vllm本地部署)

[ ] WebSocket (后端框架)

[ ] HTML/CSS/WebSocket (前端页面)


---

只使用train数据不使用test数据.

一共要得到五个数据，分别是`立场倾向判断`, `事件敏感程度`, `事件严重程度`, `事件紧急程度`, `事件影响范围`

`立场倾向判断`, `事件敏感程度` 从立场检测_train的数据算，对应文件夹

`weibo-detect-system/meta_data/立场检测_train`

`事件严重程度` 从上面两个得到的数据根据公式计算得到

`事件紧急程度`, `事件影响范围` 从train.csv算 对应文件

`weibo-detect-system/meta_data/train.csv`

---
### 构建jsonl数据
注意脚本中path更改，已进行标注

```bash
weibo-detect-system/data_process/data_stance_train.py 
```
脚本生成立场倾向判断数据`weibo-detect-system/data_process/data_stance_train.jsonl`

```bash
weibo-detect-system/data_process/data_sense_train.py
```
脚本生成事件敏感程度`weibo-detect-system/data_process/data_sense_train_gemini.jsonl`

这里脚本生成的是content：null的内容，事件敏感程度数据仅30条，通过后处理调用gemini 2.5pro模型生成label使得content内容不为空，用于后续流程

---
### 模型微调部署

[modelscope——微调模型下载](https://www.modelscope.cn/models/dabu46/qwen2.5-0.5b-ft-stand_detect/summary)


详细信息在`weibo-detect-system/stand_qwen_ft/README.md`中



