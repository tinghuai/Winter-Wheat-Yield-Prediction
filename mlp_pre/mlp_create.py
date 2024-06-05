from pandas import read_excel
import pandas as pd
from numpy import array
from joblib import dump
from sklearn.neural_network import MLPRegressor
import numpy as np
from para import steps, pre_begin, pre_steps, hidden, activation,x_c,y_c,hidden2

def get_train_data(data_x, data_y):
    model_x, model_y = [], []
    q=0
    for i in range(0,len(data_x),8):
        j = i+8
        if j>len(data_x):
            break
        x = data_x[i: j, 0:in_dim]
        y = data_y[q, 0:out_dim]
    # i    j --->  j+t+prebegin
        model_x.append(x)
        model_y.append(y)
        q+=1
    return array(model_x), array(model_y)

adr='./model/'
train_data_adr="./data/farm_train.xlsx"
train_data=read_excel(train_data_adr) #class:'pandas.core.frame.DataFrame'

print(train_data.head())

# 去异常数据 返回x_c和y_c列对应数据
x_data,y_data=train_data[x_c],train_data[y_c]
y_data = y_data.dropna()
y_data = y_data.drop_duplicates()
x_data = x_data.iloc[:, 0: len(x_c)].values#numpy.ndarray
y_data = y_data.iloc[:, 0: len(y_c)].values

in_dim = len(x_c)
out_dim = len(y_c)


train_x,train_y = get_train_data(x_data,y_data)
q, w, e = train_x.shape
train_x = train_x.reshape(q, w * e)

model = MLPRegressor(hidden_layer_sizes=(hidden,), learning_rate="adaptive", activation=activation, verbose=True)
model.fit(train_x,train_y)
dump(model, './model/model1.pkl')

# for i in range(pre_steps):
#     print(i + 1)
#     train_x, train_y = get_train_data(x_data, y_data, i)
#     q, w, e = train_x.shape
#     train_x = train_x.reshape(q, w * e)
#     print(train_x.shape)
#
#     if hidden2 == 0:
#         model = MLPRegressor(hidden_layer_sizes=(hidden,), learning_rate="adaptive",activation=activation, verbose=True)
#     else:
#         model = MLPRegressor(hidden_layer_sizes=(hidden,hidden2), learning_rate="adaptive", activation=activation,
#                              verbose=True)
#     model.fit(train_x, train_y)
#     dump(model, './model/model{}.pkl'.format(i))