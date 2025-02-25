import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

# unique_teams 是在训练模型时从 cls_result.json 文件中读取并处理得到的，它包含了所有不同的团队名称。需要重新加载这个信息才能正确初始化模型。
def load_teams(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df['分配團隊'].unique()

# 1. 加载模型和分词器
model_dir = r"../trained_model"  # 模型保存路径
model_path = os.path.join(model_dir, 'it_request_classification_model.pth')
unique_teams = load_teams('../file/cls_result.json')


'''
# 初始化 BERT 模型
BertForSequenceClassification：这是 transformers 库中的一个类，专门设计用于序列分类任务（如对文本进行分类）。
from_pretrained：此函数加载名为“bert-base-chinese”的预训练 BERT 模型。“chinese”表示此模型是在中文文本上训练的。
num_labels：此参数设置为唯一团队的数量（从 len(unique_teams) 获得）。它告诉模型需要分类到多少个不同的类别。
'''
model = BertForSequenceClassification.from_pretrained('../file/source/bert-base-chinese', num_labels=len(unique_teams)) #num_labels 需要和训练时保持一致
'''
# 将已训练模型的权重（存储在 model_path 中）加载到 BERT 模型结构中。
torch.load：此函数从文件中加载保存的模型权重。
load_state_dict：此函数使用加载的权重更新 BERT 模型的参数。
'''
model.load_state_dict(torch.load(model_path))
'''
# 此行将模型设置为“评估模式”。这很重要，因为它会停用训练期间使用的某些层或行为（如 dropout），而这些层或行为在推理（进行预测）时是不需要的。
model.train() 用于训练模型，让模型学习数据。
model.eval() 用于评估模型或使用模型进行预测，此时模型的参数不会被改变。
'''
model.eval()  # 设置为评估模式
'''
# BERT 模型的分词器
tokenizer: 这是一个变量，用于存储创建的分词器对象。之后你会使用这个 tokenizer 来准备你的输入文本，使其能够被 BERT 模型理解。
BertTokenizer: 这是来自 transformers 库的一个类，表示一个 BERT 分词器。分词器对于自然语言处理 (NLP) 至关重要，因为它们将文本分解成更小的单元（称为标记或词元），以便模型能够理解。
from_pretrained(): 这是 BertTokenizer 类的一个方法。它用于从指定位置加载一个预训练的分词器。这比从头开始训练一个分词器要快得多，也更有效率。
'''
tokenizer = BertTokenizer.from_pretrained('../file/source/bert-base-chinese')

# 2. 预测函数
def predict_team(requirement):
    team_to_label = {team: label for label, team in enumerate(unique_teams)}
    label_to_team = {label: team for team, label in team_to_label.items()}
    """
    编码需求: 使用 tokenizer（之前已加载）来处理 requirement 文本。
    1. 分词： tokenizer 将 requirement 字符串分解成称为标记的更小单元，模型可以理解这些单元。
    2. 填充： padding=True 确保所有输入序列具有相同的长度，方法是在必要时添加特殊的填充标记。深度学习模型通常需要这样做。
    3. 截断： truncation=True 如果输入序列超过模型可以处理的最大长度，则会将其缩短。
    4. return_tensors='pt'： 这指定编码输出应该是 PyTorch 张量（适合模型）。
    """
    # encoded_requirement 是一个字典，它包含了经过编码的 IT 请求信息。这个字典可能包含多个键值对
    encoded_requirement = tokenizer([requirement], padding=True, truncation=True, return_tensors='pt')
    '''
    进行预测:
    with torch.no_grad():： 此上下文管理器临时禁用梯度计算。它在推理（预测）期间使用，因为我们不需要更新模型的权重。
    output = model(**encoded_requirement):  这一行将encoded_requirement输入到预训练的model` 以获取其预测。 ** 用于将字典解包为关键字参数，这些都是模型需要的输入。相当于model()
    prediction = torch.argmax(output.logits, dim=1).item()：
        output.logits 包含每个可能的 IT 团队的原始分数。
        torch.argmax(..., dim=1) 找到得分最高的索引（标签），表示预测的团队。
        .item() 将预测提取为单个 Python 数字。
    '''
    with torch.no_grad():
        output = model(**encoded_requirement)
        prediction = torch.argmax(output.logits, dim=1).item()
    return label_to_team[prediction]  # 使用 label_to_team 将数字标签转换回团队名称

# 3. 加载标签映射 (重要!)
import json
with open('../file/cls_result.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

unique_teams = df['分配團隊'].unique()
'''
创建一个名为 team_to_label 的字典
team_to_label: 为每个从unique_teams中的团队打上数字标签. 产生团队名为键,对应的数字标签作为值
label_to_team: 与上面相反,  产生团队名对应数字标签为键, 团队名作为值
'''
# team_to_label = {team: label for label, team in enumerate(unique_teams)}
# label_to_team = {label: team for team, label in team_to_label.items()}

# 4. 调用
requirement = r"申请开通新同事Decher内联网(Intranet)和Oracle权限, 权限设置参考Hugo chen"  # 替换为您想要分类的文档内容
predicted_team = predict_team(requirement)
print(f"IT Request：{requirement}" ,f"预测分配团队：{predicted_team}")

requirement2 = r"请协助调整监控权限并取消原有权限，变动明细如下：decher-12207天环13580416184曾德祥-12133-12345北京路15013242103"
predicted_team2 = predict_team(requirement2)
print(f"IT Request：{requirement2}" ,f"预测分配团队：{predicted_team2}")

requirement3 = r"申请开通权限给Decher, 权限包括:报表平台实时销售额"
predicted_team3 = predict_team(requirement3)
print(f"IT Request：{requirement3}" ,f"预测分配团队：{predicted_team3}")

requirement4 = r"开通BI报表平台权限"
predicted_team4 = predict_team(requirement4)
print(f"IT Request：{requirement4}" ,f"预测分配团队：{predicted_team4}")

requirement5 = r"1. 开通邮箱给DecherZeng; 2.加入组别erpsc@maxims.com.hk"
predicted_team5 = predict_team(requirement5)
print(f"IT Request：{requirement5}" ,f"预测分配团队：{predicted_team5}")

requirement6 = r"修改<“>美心中国主页>主页连接为链接改为：https://aaabbcc.com"
predicted_team6 = predict_team(requirement6)
print(f"IT Request：{requirement6}" ,f"预测分配团队：{predicted_team6}")

requirement7 = r"申请开通以下路径权限:\\itgzfs01\ITShare\aabbccaaaaaaa"
predicted_team7 = predict_team(requirement7)
print(f"IT Request：{requirement7}" ,f"预测分配团队：{predicted_team7}")

requirement8 = r"申请开通VPN帐号, 给供应商新天天"
predicted_team8 = predict_team(requirement8)
print(f"IT Request：{requirement8}" ,f"预测分配团队：{predicted_team8}")

requirement9 = r"申请在09123仓新增subinventory福食, 参考09999仓下已有的ABC子仓"
predicted_team9 = predict_team(requirement9)
print(f"IT Request：{requirement9}" ,f"预测分配团队：{predicted_team9}")

requirement10 = r"测试"
predicted_team10 = predict_team(requirement10)
print(f"IT Request：{requirement10}" ,f"预测分配团队：{predicted_team10}")