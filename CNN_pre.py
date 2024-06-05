import time
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import array, read_excel
from plotly import graph_objects as go
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
import pandas as pd
from osgeo import gdal
import cv2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
from torch import nn

# 定义模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            # (-1, 5, 577, 375)
            # (-1, 1)
            nn.Conv2d(in_channels=5,out_channels=32,kernel_size=5,stride=1,padding=1),   #输入通道数=5（5个波段），输入图像32*32，卷积核3*3，步长=1，填充=2
            nn.MaxPool2d(8),
            nn.Conv2d(32, 32, 3, 1, 2),
            nn.MaxPool2d(8),
            # nn.Conv2d(32, 64, 3, 1, 2),
            # nn.MaxPool2d(8),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 读图像文件
def read_img(filename):
    dataset = gdal.Open(filename)  # 打开文件
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
    im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    del dataset  # 关闭对象dataset，释放内存
    # return im_width, im_height, im_proj, im_geotrans, im_data,im_bands
    return [im_proj, im_geotrans, im_data, im_width, im_height, im_bands]
# # 遥感影像的存储
# # 写GeoTiff文件
# def write_img(self, filename, im_proj, im_geotrans, im_data):
#     # 判断栅格数据的数据类型
#     if 'int8' in im_data.dtype.name:
#         datatype = gdal.GDT_Byte
#     elif 'int16' in im_data.dtype.name:
#         datatype = gdal.GDT_UInt16
#     else:
#         datatype = gdal.GDT_Float32
#     print(len(im_data.shape))
#
#     # 判读数组维数
#     if len(im_data.shape) == 3:
#         # 注意数据的存储波段顺序：im_bands, im_height, im_width
#         im_bands, im_height, im_width = im_data.shape
#     else:
#         im_bands, (im_height, im_width) = 1, im_data.shape
#
#     # 创建文件时 driver = gdal.GetDriverByName("GTiff")，数据类型必须要指定，因为要计算需要多大内存空间。
#     driver = gdal.GetDriverByName("GTiff")
#     dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
#
#     dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
#     dataset.SetProjection(im_proj)  # 写入投影
#
#     if im_bands == 1:
#         dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
#     else:
#         for i in range(im_bands):
#             dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
#
#     del dataset

def train_step(model, train_features, test_features, train_labels, test_labels):
    # 正向传播求损失
    train_predictions = model.forward(train_features)
    train_loss = loss_function(train_predictions, train_labels)
    test_predictions = model.forward(test_features)
    test_loss = loss_function(test_predictions, test_labels)
    # 反向传播求梯度
    train_loss.backward()
    # 参数更新
    optimizer.step()
    optimizer.zero_grad()
    return train_loss.item(), test_loss.item()
file_dir = "./modis_city/anyang/"
field = pd.read_csv('./anyang.csv', encoding='utf-8')
data = field.drop(["年份"], axis=1)
field_data = np.array(data)
# print(field_data.shape)
test_data_size = 31  # 每年31张图片
years = 15
train_data = []
test_data = []
modis_data = []
y_true_all = []
y_pre_all = []
file_list=[]
data = []
i = 0
year = 0
# 构建迭代器
epochs = 1
batch_size = 1
# for filename in file_list:
#     i+=1
#     modis_data = read_img(file_dir+filename)[2]  # run.read_img(file_list[0])返回第一个数据的五个内含数据
#     # class类中返回值第3个值是data
#     data.append(modis_data)
#     if i >30:
#         i=0
#         data = np.array(data).astype(np.float16)
#         np.save("npdata/modis_data_np_"+str(year), data)
#         year += 1
#         data = []
Y = np.zeros([15 * 31, 1])    #生成15*31行，1列的零矩阵（y值先全部填充为0）
for i in range(15):
    for j in range(31):
        Y[i * 31 + j] = j / 30 * field_data[i]
print(Y.shape)#生成15*31行，1列的零矩阵（y值先全部填充为0）
#生成15*31行，1列的零矩阵（y值先全部填充为0）

for test_year in range(15):  # 按年进行划分，一共15年
    # cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 查看第一个batch
    # x, y = next(iter(train_dataloader))
    # print(x.shape)
    # print(y.shape)
    model = CNN().to(device)
    loss_function = nn.L1Loss()  # MAE(平均绝对误差）
    # loss_function = nn.MSELoss()      #MSE（均方误差）
    # loss_function = sqrt(...)          #RMSE(均方根误差）
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)  # 更新学习率
    # 测试一个batch
    # train_features, train_labels = next(iter(train_dataloader))  # next() 返回迭代器的下一个项目，next() 函数要和生成迭代器的iter() 函数一起使用
    # test_features, test_labels = next(iter(test_dataloader))
    # train_loss = train_step(model, train_features, test_features, train_labels, test_labels)
    print('第{}年训练结束，开始下一轮训练'.format(test_year))
    train_losshistory = []
    test_losshistory = []
    test_x = np.load("npdata/modis_data_np_" + str(test_year) + ".npy")
    test_y = Y[test_data_size * test_year:test_data_size * (test_year + 1)]
    # test_y = Y[test_year]
    test_X = torch.tensor(test_x, dtype=torch.float).to(device)
    test_Y = torch.tensor(test_y, dtype=torch.float).to(device)
    test_dataset = TensorDataset(test_X, test_Y)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    for epoch in range(epochs):
        list_loss1 = []
        list_loss2 = []
        start_time = time.time()
        for part in range(15):
            if part != test_year:
                print('一共15年数据，将第{}年作为测试集训练结束'.format(part))
                train_x = np.load("npdata/modis_data_np_" + str(part) + ".npy")
                train_y = Y[test_data_size * part:test_data_size * (part + 1)]
                # 转化成 tensor->(batch_size, seq_len, feature_size)
                train_X = torch.tensor(train_x, dtype=torch.float).to(device)
                train_Y = torch.tensor(train_y, dtype=torch.float).to(device)
                train_dataset = TensorDataset(train_X, train_Y)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
                for train_features, train_labels in train_dataloader:
                    test_features, test_labels = next(iter(test_dataloader))
                    loss1, loss2 = train_step(model, train_features, test_features, train_labels, test_labels)
                    list_loss1.append(loss1)
                    list_loss2.append(loss2)
        end_time = time.time()
        print("运行时间：" + str(end_time - start_time) + "秒")
        train_loss = np.mean(list_loss1)
        test_loss = np.mean(list_loss2)
        # if epoch >= 50:
        scheduler.step(train_loss, epoch)
        # if epoch % 10 == 0:
        print('epoch={} | loss={} '.format(epoch+1, train_loss))
        train_losshistory.append(train_loss)
        test_losshistory.append(test_loss)

    train_losshistory = np.array(train_losshistory)
    test_losshistory = np.array(test_losshistory)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i for i in range(len(train_losshistory))], y=train_losshistory, name='train_loss'))
    fig.add_trace(go.Scatter(x=[i for i in range(len(test_losshistory))], y=test_losshistory, name='test_loss'))
    fig.show()
    # %%
    # 训练集的测试
    # y_true = Y.cpu().numpy().squeeze()
    # y_pred = model.forward(X).detach().cpu().numpy().squeeze()
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(y=y_true, name='y_true'))
    # fig.add_trace(go.Scatter(y=y_pred, name='y_pred'))
    # fig.show()

    # 测试集的结果
    x_test = torch.tensor(test_x, dtype=torch.float).to(device)
    y_test = torch.tensor(test_y, dtype=torch.float).to(device)
    y_true = y_test.cpu().numpy().squeeze()
    y_pre = model.forward(x_test).detach().cpu().numpy().squeeze()
    y_true_all.append(y_true)
    y_pre_all.append(y_pre)

# 15年 18市 测试结果汇总
fig = go.Figure()
fig.add_trace(go.Scatter(y=np.array(y_true_all).reshape(-1), name='y_true'))
fig.add_trace(go.Scatter(y=np.array(y_pre_all).reshape(-1), name='y_pre'))
fig.show()
