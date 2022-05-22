import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

city = '上海'
path = '..\..\dataset\data_xyz\provinceData\\' + city + '.json'  ##记得改
fr = open(path, 'r', encoding='utf-8')
json_info = fr.read()
fr.close()
data_dict = json.loads(json_info)
data = pd.DataFrame(data_dict['data'])
index = data['dateId']
data = data['confirmedCount']
value = data.values
x = []
y = []
seq = 7
for i in range(len(value) - seq):
    x.append(value[i:i + seq])
    y.append(value[i + seq])
# print(x, '\n', y)
length = len(x)
train_len = int(length * 0.7)
train_x = (torch.tensor(x[:train_len]).float() / 100000.).reshape(-1, seq, 1)
train_y = (torch.tensor(y[:train_len]).float() / 100000.).reshape(-1, 1)
test_x = (torch.tensor(x[train_len:]).float() / 100000.).reshape(-1, seq, 1)
test_y = (torch.tensor(y[train_len:]).float() / 100000.).reshape(-1, 1)


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
epoch_num = 400
for epoch in range(epoch_num):
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
prediction = list(((model(train_x).data.reshape(-1)) * 100000)) + list(((model(test_x).data.reshape(-1)) * 100000))
index = index[:len(value) - seq]
plt.plot(value[:length], label='True Value')
plt.plot(prediction[:train_len + 1], label='LSTM fit')
plt.plot(np.arange(train_len, length, 1), prediction[train_len:length], label='LSTM pred')
print('true_values: ', value[:length])
print('predict_values: ', prediction)
true_confirm = list(value[:length])
predict_confirm = prediction
index = list(index)
predict_confirm = [float(x) for x in predict_confirm]
result = pd.DataFrame({'date': index, 'true_confirm': true_confirm, 'predict_confirm': predict_confirm})
print(result)
result.to_csv('..\..\dataset\data_predict\LSTM\\' + city + '预测结果.csv')
plt.legend(loc='best')
plt.title('Active infections prediction')
plt.xlabel('Day')
plt.ylabel('Active Cases')
plt.savefig(city + "预测.jpg")
plt.show()

print(f"根均方误差(RMSE)：{np.sqrt(mean_squared_error(prediction, value[:len(prediction)]))}")
