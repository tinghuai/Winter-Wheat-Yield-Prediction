steps = 10 # 向前分析60s
pre_begin = 30  # 超前预测30s
pre_steps = 1  # 向后预测2s
activation='relu'
# activation='tanh'
# activation = 'logistic'
hidden =  100
hidden2 = 0


x_c=['平均蒸散量','最高温度','最低温度','干旱指数','降水量','土壤湿度','面积']
y_c=['产量']


