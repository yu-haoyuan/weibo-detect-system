# 微调qwen2.5 0.5b

数据集里面唯一数据量足够大的就是`立场倾向判断`这个任务，而且适合微调
根据任务描述，输出为-1， 0， 1三分类代表反对（极端）， 中立， 支持（极端）
为了简化任务就不进行五分类，只进行三分类
数据集即为`weibo-detect-system/data_process`中构建的`weibo-detect-system/data_process/data_stance_train.jsonl`

使用llama-factory进行微调，共1w5左右条数据(https://github.com/hiyouga/LLaMA-Factory)

### 1.配置llama-factory
```
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

### 2.通过huggingface镜像下载模型, (https://hf-mirror.com/)

这里下载Qwen2.5-0.5B-Instruct,Instruct版本微调过，能力好点（https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct）
```
huggingface-cli download --resume-download Qwen/Qwen2.5-0.5B-Instruct --local-dir /your/path
```

### 3.进行微调
##### 3.1 改`LLaMA-Factory/data/dataset_info.json`文件

拉到最下面追加写入这一整块，注意是追加，所以需要在上一个`}`结束的地方加一个逗号，注意整个大括号的关闭
```
  "stan_train": {   #"stan_train"为自定义数据集命名，这里代表立场数据
    "file_name": "/data/xiaobu/LLaMA-Factory/mydata/stan_data/stan_train.jsonl",  #要训练的数据集地址
    "formatting": "sharegpt",
  "columns": {
    "messages": "messages"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system"
  }
    }
```

##### 3.2 写train yaml文件执行训练

可以创建一个my_yaml文件夹，创建新文件：`LLaMA-Factory/myyaml/qwen2.5_0.5b_ins_train.yaml`

内容为
```
model_name_or_path: /data/xiaobu/LLaMA-Factory/mymodel/Qwen2.5-0.5B-Instruct #第二步里面--local-dir /your/path存放的位置
trust_remote_code: true

stage: sft
do_train: true
lora_rank: 8
lora_target: all

dataset: stan_train #和3.1中json更改的key："stan_train"对齐
template: qwen #如果是别的小模型，这里要改
finetuning_type: lora
output_dir: /data/xiaobu/LLaMA-Factory/myoutputs/qwen2.5-0.5b-ft-2/ #你的输出地址
num_train_epochs: 2
per_device_train_batch_size: 16 #这里要改小，根据自己的显存能力，笔记本显卡一般4可以
gradient_accumulation_steps: 4
learning_rate: 2e-5
logging_steps: 10
save_steps: 200
save_total_limit: 2
fp16: true
plot_loss: true
```

创建完毕后，执行
```bash
llamafactory-cli train LLaMA-Factory/myyaml/qwen2.5_0.5b_ins_train.yaml
```

##### 3.3 写infer yaml文件执行推理

创建新文件：`LLaMA-Factory/myyaml/qwen2.5_0.5b_ins_infer.yaml`
内容为
```
model_name_or_path: /data/xiaobu/LLaMA-Factory/mymodel/Qwen2.5-0.5B-Instruct
adapter_name_or_path: /data/xiaobu/LLaMA-Factory/myoutputs/qwen2.5-0.5b-ft-2
template: qwen
infer_backend: huggingface  # choices: [huggingface, vllm, sglang]
trust_remote_code: true
```
如果可以正常运行，就可以合并lora了

##### 3.4 写merge yaml文件导出合并后模型

创建新文件：`LLaMA-Factory/myyaml/qwen_merge_lora.yaml`
内容为：
```
### examples
### model
model_name_or_path: /data/xiaobu/LLaMA-Factory/mymodel/Qwen2.5-0.5B-Instruct
adapter_name_or_path: /data/xiaobu/LLaMA-Factory/myoutputs/qwen2.5-0.5b-ft-2
template: qwen
finetuning_type: lora

### export
export_dir: /data/xiaobu/LLaMA-Factory/myoutputs/qwen2.5-0.5b-ft-merged
export_size: 2 #分两个切片导入
export_device: cpu #可以cpu推理
export_legacy_format: false
```

##### 3.5 对导出的模型进行推理

首先处理数据，把原数据改成无标签的待推理数据

这是在最开始数据处理得到的`weibo-detect-system/data_process/data_stance_train.jsonl`立场数据集

```bash
python weibo-detect-system/stand_qwen_ft/data/split_ft.py
python weibo-detect-system/stand_qwen_ft/data/make_test_stan_jsonl.py
```
第一个脚本把数据分为train test val
第二个脚本把assistant的content重置为空用于推理

准备好数据后执行
```bash
python weibo-detect-system/stand_qwen_ft/infer.py
```

##### 4下载微调好的模型
（https://www.modelscope.cn/models/dabu46/qwen2.5-0.5b-ft-stand_detect/summary）
自定义modelscope模型下载路径
```bash
export MODELSCOPE_CACHE='您希望的下载路径'
```