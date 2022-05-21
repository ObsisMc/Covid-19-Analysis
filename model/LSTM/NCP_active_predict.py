import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import json
# 数据预处理
import pandas as pd
path = '..\..\dataset\data_xyz\provinceData\上海市.json' ##记得改
fr = open(path, 'r', encoding='utf-8')
json_info = fr.read()
fr.close()
data_dict = json.loads(json_info)
data = pd.DataFrame(data_dict['data'])
data =data['confirmedCount']
value =data.values
print(len(value))
x = []
y = []
seq = 3
for i in range(len(value)-seq):
    x.append(value[i:i+seq])
    y.append(value[i+seq])
#print(x, '\n', y)
print(len(x))   # 67 828
length =len(x)
train_len =int(length*0.7)
train_x = (torch.tensor(x[:train_len]).float()/100000.).reshape(-1, seq, 1)
train_y = (torch.tensor(y[:train_len]).float()/100000.).reshape(-1, 1)
test_x = (torch.tensor(x[train_len:]).float()/100000.).reshape(-1, seq, 1)
test_y = (torch.tensor(y[train_len:]).float()/100000.).reshape(-1, 1)
print(len(train_x))
print(len(test_x))

# 模型训练
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
        self.linear = nn.Linear(16 * seq, 1)
    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, 16 * seq)
        x = self.linear(x)
        return x

# 模型训练
model = LSTM()
optimzer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_func = nn.MSELoss()
model.train()

for epoch in range(400):
    output = model(train_x)
    loss = loss_func(output, train_y)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    if epoch % 20 == 0:
        tess_loss = loss_func(model(test_x), test_y)
        print("epoch:{}, train_loss:{}, test_loss:{}".format(epoch, loss, tess_loss))

# 模型预测、画图
model.eval()
prediction = list((model(train_x).data.reshape(-1))*100000) + list((model(test_x).data.reshape(-1))*100000)
plt.plot(value, label='True Value')
plt.plot(prediction[:train_len+1], label='LSTM fit')
plt.plot(np.arange(train_len, length, 1), prediction[train_len:length], label='LSTM pred')
print(value)
print(prediction)
plt.legend(loc='best')
plt.title('Active infections prediction')
plt.xlabel('Day')
plt.ylabel('Active Cases')
plt.show()
