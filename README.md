1. 从https://huggingface.co/google-bert/bert-base-chinese/tree/main 下载预训练模型`bert-base-chinese`
2. 代码在path: `..\code`
- 先用`data_cleaner.py`清洗数据, 清洗前需要把it request放在`../file/itrequest.xlsx`; 清洗完的文件放在`../file/cls_result.json`; 需要清除的特殊字符指定在`../file/specify_char.json`
. 使用`Model_Training.py`训练模型, 训练前需确保清理完的训练数据集放在`../file/cls_result.json`; `bert-base-chinese`模型放置在`../file/source/bert-base-chinese`; 完成训练的模型放置在'../trained_model'
- 使用`itrequest_predict.py`进行预测, 需要预测的内容在py文件的末尾
