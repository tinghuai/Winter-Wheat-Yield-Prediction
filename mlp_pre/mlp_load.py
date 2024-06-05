from pandas import read_excel
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from math import sqrt
from matplotlib import pyplot, pyplot as plt
from numpy import array
import numpy as np
import pandas as pd
from joblib import load
from matplotlib.font_manager import FontProperties
# from tensorflow.python.keras.models import load_model
from para import steps, pre_begin, pre_steps, hidden, activation,x_c,y_c,hidden2
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=24)
np.set_printoptions(suppress=True)

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
in_dim = len(x_c)
out_dim = len(y_c)
adr='./model/'
val_data_adr="./data/farm_val.xlsx"
val_data=read_excel(val_data_adr) #class:'pandas.core.frame.DataFrame'

print(val_data.head())


val_x,val_y = val_data[x_c],val_data[y_c]
val_y = val_y.dropna()
val_y = val_y.drop_duplicates()
val_x = val_x.iloc[:, 0: len(x_c)].values
val_y = val_y.iloc[:, 0: len(y_c)].values
out_dim =len(y_c)

val_x,val_y = get_train_data(val_x,val_y)
q, w, e = val_x.shape
val_x = val_x.reshape(q, w * e)

model_path = r'./model/model1.pkl'
#model = load_model(model_path)
model=load(model_path)
pre_y = model.predict(val_x)

_pre = pre_y.reshape(-1, out_dim)
_val = val_y.reshape(-1,out_dim)
plt.tick_params(labelsize=20)
plt.scatter(np.arange(len(_pre)), _pre, s=20, label='predict', color='red')
plt.scatter(np.arange(len(_val)), _val, s=20, label='val data', color='blue')
plt.title('15、16、17年数据预测产量', fontproperties=font_set)
plt.savefig('./val_pic.png')
rmse = sqrt(mean_squared_error(pre_y, val_y))
mae = mean_absolute_error(pre_y, val_y)

print('预测产量 RMSE: %.4f \n' % rmse)

plt.clf()



