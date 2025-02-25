import json
import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. 数据准备
# 从 JSON 文件加载数据
with open('../file/cls_result.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 2. 数据预处理
# 使用 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('../file/source/bert-base-chinese')

# 将文本数据转换为 BERT 模型可接受的格式
def encode_data(texts, labels):
    encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    encoded_labels = torch.tensor(labels)
    return encoded_texts, encoded_labels

# 将“分配团队”转换为数字标签
unique_teams = df['分配團隊'].unique()
team_to_label = {team: label for label, team in enumerate(unique_teams)}
label_to_team = {label: team for team, label in team_to_label.items()}

df['label'] = df['分配團隊'].map(team_to_label)

# 3. 数据集划分
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 4. 模型训练
# 加载预训练的 BERT 模型，并添加一个分类层
model = BertForSequenceClassification.from_pretrained('../file/source/bert-base-chinese', num_labels=len(unique_teams))

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 训练模型
def train_model(model, train_df, optimizer, loss_fn, epochs=12):
    for epoch in range(epochs):
        for i in range(0, len(train_df), 16):  # batch size = 16
            batch_df = train_df[i:i+16]
            texts = batch_df['需求內容'].tolist()
            labels = batch_df['label'].tolist()

            encoded_texts, encoded_labels = encode_data(texts, labels)  # 调用外部的 encode_data 函数

            optimizer.zero_grad()
            outputs = model(**encoded_texts, labels=encoded_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} Loss: {loss.item()}")

train_model(model, train_df, optimizer, loss_fn)

# 5. 模型评估
def evaluate_model(model, test_df):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_df), 16):
            batch_df = test_df[i:i+16]
            texts = batch_df['需求內容'].tolist()
            labels = batch_df['label'].tolist()
            encoded_texts, encoded_labels = encode_data(texts, labels)

            outputs = model(**encoded_texts)
            predictions = torch.argmax(outputs.logits, dim=1)
            total += encoded_labels.size(0)
            correct += (predictions == encoded_labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy: {accuracy}")

evaluate_model(model, test_df)

# 6. 模型保存
model_dir = r"../trained_model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)  # 如果目录不存在，则创建
model_path = os.path.join(model_dir, 'it_request_classification_model.pth')

torch.save(model.state_dict(), model_path)

# # 7. 预测函数
# def predict_team(requirement):
#     encoded_requirement = tokenizer([requirement], padding=True, truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         output = model(**encoded_requirement)
#         prediction = torch.argmax(output.logits, dim=1).item()
#     return label_to_team[prediction]
#
# # 示例
# requirement = "开通邮箱权限"
# predicted_team = predict_team(requirement)
# print(f"预测分配团队：{predicted_team}")