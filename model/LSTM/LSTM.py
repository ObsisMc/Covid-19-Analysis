import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import  torch.nn as nn
from torch import  optim

import json
import sklearn

RANDOM_SEED = 16
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data.iloc[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


def data_preprocess():
    path = 'D:\我的大学\大三下\数据挖掘\project\Covid-19-Analysis\dataset\data_xyz\provinceData\上海市.json' ##记得改
    fr = open(path, 'r', encoding='utf-8')
    json_info = fr.read()
    fr.close()
    data_dict = json.loads(json_info)
    data = pd.DataFrame(data_dict['data'])
    data = data[['confirmedIncr', 'dateId']]
    data['dateId'] = data['dateId'].astype(str)
    data['dateId'] = pd.to_datetime(data['dateId'], format='%Y%m%d')
    data.index = data['dateId']
    data = data.set_index('dateId', drop=True)
    data_size = int(len(data)*0.7)
    test_set = data[:-data_size]
    train_set = data[-data_size:]

    """
    scaler = MinMaxScaler()
    train_set = scaler.fit_transform(train_set)
    test_set = scaler.fit_transform(test_set)
    """
    seq_length = 7
    """
    train_set=pd.DataFrame(train_set)
    test_set =pd.DataFrame(test_set)
    """
    X_train, y_train = create_sequences(train_set, seq_length)
    X_test, y_test = create_sequences(test_set, seq_length)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()



    """
    scaler = MinMaxScaler()
    scaler = scaler.fit(np.expand_dims(train_set, axis=1))

    train_set = scaler.transform(np.expand_dims(train_set, axis=1))
    test_set = scaler.transform(np.expand_dims(test_set, axis=1))
    print(train_set)
    """
    return X_train,y_train, X_test, y_test

class lstm(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(lstm, self).__init__()

        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=0.5
        )

        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )

    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len, -1),
            self.hidden
        )
        last_time_step = \
            lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred


def train_model(model, x_train, y_train,test_data=None,test_labels=None):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100
    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)
    for t in range(num_epochs):
        model.reset_hidden_state()
        y_pred = model(x_train)
        loss = loss_fn(y_pred.float(), y_train)
        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(x_test)
                test_loss = loss_fn(y_test_pred.float(), y_test)
            test_hist[t] = test_loss.item()
            if t % 10 == 0:
                print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
        elif t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')
        train_hist[t] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval(), train_hist, test_hist
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = data_preprocess()
    model = lstm( n_features=1, n_hidden=512, seq_len=7,  n_layers=2)

    model, train_hist, test_hist = train_model(model, x_train,y_train,x_test,y_test)