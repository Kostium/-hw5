import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt


train_data_path = 'dataset/train.txt'
test_data_path = 'dataset/test_without_label.txt'
train_data = pd.read_csv(train_data_path, delimiter=',', header=0)
test_data = pd.read_csv(test_data_path, delimiter=',', header=0)
# 划分训练集和验证集
train_df, val_df = train_test_split(train_data, test_size=0.2, random_state=42)
# print(train_df)
# print(train_df['tag'].dtype)

data_folder = 'dataset/data'

text_data_list_train = [] # 创建一个空列表用于存放文本数据
image_data_list_train = [] # 创建一个空列表用于存放图像数据
text_data_list_val = []
image_data_list_val = []
true_labels_val = []
text_data_list_test = []
image_data_list_test = []

# 映射标签为整数
tag_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
train_df['tag_encoded'] = train_df['tag'].map(tag_mapping)
# 定义训练集的标签
train_labels = torch.tensor(train_df['tag_encoded'].values, dtype=torch.long)

def load_data(guid, state):
    # 加载文本数据
    file_path = os.path.join(data_folder, f'{guid}.txt')
    image_file_path = os.path.join(data_folder, f'{guid}.jpg')
    with open(file_path, 'r', encoding='ISO-8859-1') as text_file:
        text_data = text_file.read()
        if state == 'is_train':
            text_data_list_train.append(text_data)
            image_data_list_train.append(Image.open(image_file_path).convert('RGB'))
        elif state == 'is_val':
            text_data_list_val.append(text_data)
            image_data_list_val.append(Image.open(image_file_path).convert('RGB'))
        elif state == 'is_test':
            text_data_list_test.append(text_data)
            image_data_list_test.append(Image.open(image_file_path).convert('RGB'))


# 遍历 train_df(训练集) ，加载对应的文本和图片数据
for index, row in train_df.iterrows():
    current_guid, current_tag = row['guid'], row['tag']
    load_data(current_guid, state = 'is_train')

# 遍历 val_df(验证集) ，加载对应的文本和图片数据
for index, row in val_df.iterrows():
    current_guid_val, current_tag_val = row['guid'], row['tag']
    load_data(current_guid_val, state = 'is_val')
    true_labels_val.append(tag_mapping[current_tag_val])

# 遍历 test_data(测试集) ，加载对应的文本和图片数据
for index, row in test_data.iterrows():
    current_guid_test, current_tag_test = int(row['guid']), row['tag']
    load_data(current_guid_test,  state = 'is_test')
print("数据预处理已完成")

# 使用 TfidVectorizer 进行文本向量化
vectorizer = TfidfVectorizer()
text_vectorizer_train = vectorizer.fit_transform(text_data_list_train)
text_features_train = torch.Tensor(text_vectorizer_train.todense())
text_vectorizer_val = vectorizer.transform(text_data_list_val)
text_features_val = torch.Tensor(text_vectorizer_val.todense())
text_vectorizer_test = vectorizer.transform(text_data_list_test)
text_features_test = torch.Tensor(text_vectorizer_test.todense())


# 定义图像数据的变换
image_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # LeNet使用32x32的图像
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 标准化到 [-1, 1]
])

# 将图像数据转换为 PyTorch 张量
image_data_tensor = torch.stack([image_transform(image) for image in image_data_list_train])
image_data_tensor_val = torch.stack([image_transform(image) for image in image_data_list_val])
image_data_tensor_test = torch.stack([image_transform(image) for image in image_data_list_test])

# 定义LeNet架构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)  # 3是类别数量
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)  # 使用log_softmax作为激活函数

# 创建LeNet模型的实例
lenet_model = LeNet()

# 使用LeNet模型提取验证集和测试集图像特征
image_features = lenet_model(image_data_tensor) # 使用LeNet模型提取图像特征
image_features = image_features.view(image_features.size(0), -1) # 将图像特征展平
image_features_val = lenet_model(image_data_tensor_val)
image_features_val = image_features_val.view(image_features_val.size(0), -1)
image_features_test = lenet_model(image_data_tensor_test)
image_features_test = image_features_test.view(image_features_test.size(0), -1)

# 合并验证集和测试集文本特征和图像特征
merged_features = torch.cat([text_features_train, image_features], dim=1) # 合并文本特征和图像特征
merged_features_tensor = torch.Tensor(merged_features)# 将合并后的特征转为 PyTorch 张量
merged_features_val = torch.cat([text_features_val, image_features_val], dim=1)
merged_features_tensor_val = torch.Tensor(merged_features_val)
merged_features_test = torch.cat([text_features_test, image_features_test], dim=1)
merged_features_tensor_test = torch.Tensor(merged_features_test)

# 定义一个简单的多模态模型
class MultimodalModel(nn.Module):
    def __init__(self, input_size):
        super(MultimodalModel, self).__init__()
        # 输入层
        self.fc1 = nn.Linear(input_size, 512)
        # 隐藏层
        self.fc2 = nn.Linear(512, 256)
        # 输出层
        self.fc3 = nn.Linear(256, num_classes)  # num_classes 是输出类别数量

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


input_size = merged_features.shape[1]  # 根据合并后的特征的形状确定
num_classes = 3  # 类别数量
# 创建多模态模型的实例
multimodal_model = MultimodalModel(input_size)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(multimodal_model.parameters(), lr=0.0001)

# 训练循环
num_epochs = 10
# 记录损失值
train_losses = []
for epoch in range(num_epochs):
    for index in range(0, len(merged_features_tensor), 64):
        # 提取输入和标签
        inputs = merged_features_tensor[index:index+64]
        labels = train_labels[index:index+64]
        # 清除之前的梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = multimodal_model(inputs)
        loss = criterion(outputs, labels)
        # 反向传播和优化
        loss.backward(retain_graph=True)
        optimizer.step()
        # 记录训练损失
        train_losses.append(loss.item())
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 绘制训练和验证损失图
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 预测标签
with torch.no_grad():
    multimodal_model.eval()
    outputs_val = multimodal_model(merged_features_tensor_val)
    _, predicted_val = torch.max(outputs_val, 1)

# 计算准确率
correct_val = (predicted_val == torch.tensor(true_labels_val)).sum().item()
total_val = len(true_labels_val)
accuracy_val = correct_val / total_val
print(f'Validation Accuracy: {accuracy_val}')

# 打印模型在验证集上的预测结果
# for i in range(len(true_labels_val)):
#     guid_val = val_df.iloc[i]['guid']
#     true_label_val = val_df.iloc[i]['tag']
#     predicted_label_val = list(tag_mapping.keys())[predicted_val[i]]
#     print(f'GUID: {guid_val}, True Label: {true_label_val}, Predicted Label: {predicted_label_val}')

#在测试集上预测结果
with torch.no_grad():
    multimodal_model.eval()
    outputs_test = multimodal_model(merged_features_tensor_test)
    _, predicted_test = torch.max(outputs_test, 1)
for i in range(len(predicted_test)):
    guid_test = test_data.iloc[i]['guid']
    predicted_label_test = list(tag_mapping.keys())[predicted_test[i]]
    print(f"{int(guid_test)},{predicted_label_test}")




