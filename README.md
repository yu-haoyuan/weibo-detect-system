# weibo-detect-system

只使用train数据不使用test数据

一共要得到五个数据，分别是`立场倾向判断`, `事件敏感程度`, `事件严重程度`, `事件紧急程度`, `事件影响范围`

`立场倾向判断`, `事件敏感程度` 从立场检测_train的数据算，对应文件夹

`weibo-detect-system/meta_data/立场检测_train`

`事件严重程度` 从上面两个得到的数据根据公式计算得到

`事件紧急程度`, `事件影响范围` 从train.csv算 对应文件

`weibo-detect-system/meta_data/train.csv`

# 以下是如何构建jsonl数据
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




