###### 只使用train数据不使用test数据
*立场倾向判断* *事件敏感程度* 从立场检测_train的数据算，对应文件夹weibo-detect-system/meta_data/立场检测_train内容
*事件严重程度* 从上面两个得到的数据根据公式计算得到

*事件紧急程度* *事件影响范围* 从train.csv算 对应文件weibo-detect-system/meta_data/train.csv
'''
weibo-detect-system/data_process/data_stance_train.py 
'''
脚本生成
'''
weibo-detect-system/data_process/data_stance_train.jsonl
'''
文件