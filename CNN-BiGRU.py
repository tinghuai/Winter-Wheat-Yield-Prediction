import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import array, read_excel
from plotly import graph_objects as go
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 定义模型
class CNN_bigru(nn.Module):
    def __init__(self):
        super(CNN_bigru, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # (batch_size=1, seq_len=8, input_size=10) ---> permute(0, 2, 1)
        # (1, 10, 8)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        # (batch_size=1, out_channels=64, seq_len-2+1=7) ---> permute(0, 2, 1)
        # (1, 7, 64)
        self.bigru = nn.GRU(input_size=32, hidden_size=32,
                            num_layers=1, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bigru(x,None)
        # x= self.dropout(x)
        x = self.fc(x)
        x = x[:, -1, :]  # LSTM最后一个输出seq_len
        return x

def train_step(model, train_features,test_features, train_labels,test_labels):
    # 正向传播求损失
    train_predictions = model.forward(train_features)
    train_loss = loss_function(train_predictions, train_labels)
    test_predictions = model.forward(test_features)
    test_loss = loss_function(test_predictions,test_labels)
    # 反向传播求梯度
    train_loss.backward()
    # 参数更新
    optimizer.step()
    optimizer.zero_grad()
    return train_loss.item(),test_loss.item()

env_data = pd.read_csv('./dataVI.csv', encoding='gbk')
data = env_data.drop(["城市", "日期"], axis=1)
# array_data =np.array(data)[:,:11]
arr_data = np.array(data)
# array_data =np.concatenate([np.array(data)[:,:6],np.array(data)[:,-1:]],axis=1) #前六列+后两列
# data_nor = (arr_data - arr_data.min(axis=0))/(arr_data.max(axis=0)-arr_data.min(axis=0))
# array_data =np.array(data_nor)
array_data=np.array(arr_data)
#print(array_data.shape)
#print(array_data.shape)
test_data_size = 8
city_num = 18
years = 15
y_true_all = []
y_pre_all = []
for i in range(years):   #按年进行划分，一共15年
    train_data = []
    test_data = []
    for j in range(city_num):#每年按城市划分，一共18个城市
        city_data = array_data[j * test_data_size * years:(j + 1) * test_data_size * years] #第j个城市15年每年8个月的总数据
        train = np.concatenate([city_data[:i * test_data_size], city_data[(i + 1) * test_data_size:]], axis=0)
        train_data.append(train)   #第j个城市去掉一年，14年每年8个月的数据
        test_data.append(city_data[i * test_data_size:(i + 1) * test_data_size])#18个城市留同一年数据作为测试集
    train_data = np.array(train_data)   #训练集转为数组
    # train_data = train_data.reshape(-1, 8, 7)
    train_data = train_data.reshape(-1, 8, 11)        #（补位）维度：第一维：第一个城市，第二个城市...；第二维：每年8个月（8行数据），第三维：x,y9个因素(9列）
    train_x = train_data[:, :, :-1]      #每批训练样本（8个月（8行）所有行前8列的数据（影响因素x值））
    train_y = train_data[:, -1, -1]      #每批训练样本（8个月（8行））最后一行最后一列的数据（单位面积产量数据y值）
    #print(train_x.shape)            #（8*15*18=2160，7）
    #print(train_y.shape)            #（14*18=252，8，6）
    test_data = np.array(test_data)
    # test_data = test_data.reshape(-1, 8, 7)
    test_data = test_data.reshape(-1, 8, 11)
    test_x = test_data[:, :, :-1]
    test_y = test_data[:, -1, -1]
    #print(test_x.shape)        #（14*18=252）
    #print(test_y.shape)        #(18,8,6)
    # %%
    # cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 转化成 tensor->(batch_size, seq_len, feature_size)
    train_X = torch.tensor(train_x, dtype=torch.float).to(device)
    train_Y = torch.tensor(train_y, dtype=torch.float).to(device)
    test_X = torch.tensor(test_x, dtype=torch.float).to(device)
    test_Y = torch.tensor(test_y, dtype=torch.float).to(device)
    #print('Total datasets: ', X.shape, '-->', Y.shape)            #将训练集数据转为tensor格式
    # %%
    # 构建迭代器
    epoch = 500
    batch_size = 1
    train_dataset = TensorDataset(train_X, train_Y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    test_dataset = TensorDataset(test_X, test_Y)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    # 查看第一个batch
    #x, y = next(iter(train_dataloader))
    #print(x.shape)
    #print(y.shape)
    # torchkeras API 训练方式
    # model = torchkeras.Model(Net())
    # model.summary(input_shape=(8, 6))
    # optimizer_ExpLR = torch.optim.Adam(model.parameters(), lr=1e-4)
    # model.compile(loss_func=F.mse_loss, optimizer=optimizer_ExpLR, device=device)
    # dfhistory = model.fit(epochs=100, dl_train=dl_train, log_step_freq=20)
    # %%
    # 模型评估,损失函数图
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=dfhistory.index, y=dfhistory['loss'], name='loss'))
    # fig.show()
    # pytorch 训练的写法
    model = CNN_bigru().to(device)
    loss_function = nn.MSELoss()        #均方误差
    # loss_function = nn.L1Loss()       #MAE(平均绝对误差）
    # # loss_function = sqrt(...)       #RMSE(均方根误差）
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10, verbose=True)  #更新学习率
    train_features, train_labels = next(iter(train_dataloader))    #next() 返回迭代器的下一个项目，next() 函数要和生成迭代器的iter() 函数一起使用
    test_features, test_labels = next(iter(test_dataloader))
    train_loss = train_step(model, train_features, test_features,train_labels,test_labels)


    def train_model(model, epochs):
        global list_loss1, list_loss2, train_losshistory,test_losshistory
        train_losshistory = []
        test_losshistory = []
        for epoch in range(1, epochs + 1):
            list_loss1 = []
            list_loss2 = []
            for train_features, train_labels in train_dataloader:
                test_features, test_labels = next(iter(test_dataloader))
                loss1,loss2 = train_step(model, train_features,test_features, train_labels,test_labels)
                list_loss1.append(loss1)
                list_loss2.append(loss2)
            train_loss = np.mean(list_loss1)
            test_loss =np.mean(list_loss2)
            if epoch >=50:
                scheduler.step(train_loss, epoch)
            if epoch % 10 == 0:
                print('epoch={} | loss={} '.format(epoch, train_loss))
            train_losshistory.append(train_loss)
            test_losshistory.append(test_loss)
        train_losshistory = np.array(train_losshistory)
        test_losshistory = np.array(test_losshistory)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[i for i in range(len(train_losshistory))], y=train_losshistory, name='train_loss'))
        fig.add_trace(go.Scatter(x=[i for i in range(len(test_losshistory))], y=test_losshistory, name='test_loss'))
        fig.show()
    print('第{}年训练结束，开始下一轮训练'.format(i))
    train_model(model, epoch)
    # %%
    # 训练集的测试
    # y_true = Y.cpu().numpy().squeeze()
    # y_pred = model.forward(X).detach().cpu().numpy().squeeze()
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(y=y_true, name='y_true'))
    # fig.add_trace(go.Scatter(y=y_pred, name='y_pred'))
    # fig.show()

    # 测试集的结果
    X = torch.tensor(test_x, dtype=torch.float).to(device)
    Y = torch.tensor(test_y, dtype=torch.float).to(device)
    y_true = Y.cpu().numpy().squeeze()
    y_pre = model.forward(X).detach().cpu().numpy().squeeze()
    y_true_all.append(y_true)
    y_pre_all.append(y_pre)
# y_loss = [ abs(y_true_all[i] - y_pre_all[i]) for i in range(len( y_true_all))]
# y_loss1 = []
# for i in range(15):
#     for j in range(18):
#         y_loss1.append(y_loss[i][j])
# y_loss_all = np.mean(y_loss1)
# print(y_loss_all)

print("MAE:", mean_absolute_error(y_true_all, y_pre_all))
print("MSE:", mean_squared_error(y_true_all, y_pre_all))
print("RMSE:", sqrt(mean_squared_error(y_true_all, y_pre_all)))
# print("R²:", r2_score(y_true_all, y_pre_all))

# 15年 18市 测试结果汇总
fig = go.Figure()
fig.add_trace(go.Scatter(y=np.array(y_true_all).reshape(-1), name='y_true'))
fig.add_trace(go.Scatter(y=np.array(y_pre_all).reshape(-1), name='y_pre'))
fig.show()
