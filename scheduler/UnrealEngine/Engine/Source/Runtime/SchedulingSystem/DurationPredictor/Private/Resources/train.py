import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import AdaBoostRegressor
from joblib import dump, load

names = ['add', 'cpy', 'mul_mat', 'mul', 'rms_norm', 'rope', 'silu', 'soft_max']
k = [2, 4, 21, 3, 2, 5, 1, 8]

with open(f'measure.txt', 'w') as f:
    for name in names:
        # 读取数据
        input_list = []
        time_list = []
        count_list = []
        length = 9 if name in ['rms_norm', 'silu'] else 13
        data = []
        with open(f'output_{name}.txt', 'r') as f2:
            for line in f2:
                row = list(map(float, line.split(',')))
                if len(row) == length:
                    data.append(row)
                    if row[:-1] not in input_list:
                        input_list.append(row[:-1])
                        time_list.append(row[-1])
                        count_list.append(1)
                    else:
                        time_list[input_list.index(row[:-1])] += row[-1]
                        count_list[input_list.index(row[:-1])] += 1

        # 将数据分为特征和标签
        data = np.array(data)
        features = data[:, :-1]
        labels = data[:, -1] * 1000

        # 划分训练集和测试集
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # 线性回归模型
        # model1 = LinearRegression()
        # model1.fit(features_train, labels_train)
        # labels_pred = model1.predict(features_test)
        # mape1 = np.mean(np.abs((labels_test - labels_pred) / labels_test)) * 100
        # content = f'{name} MAPE LR : {mape1}'
        # print(content)
        # f.write(content + '\n')

        # KNN模型
        end = 24 if name == 'mul_mat' else 10
        for i in range(1, end):
            name_k = k[names.index(name)]
            if i == name_k:
                model2 = KNeighborsClassifier(n_neighbors=i) 
                model2.fit(features_train, labels_train.astype(int))
                labels_pred2 = model2.predict(features_test)
                mape2 = np.mean(np.abs((labels_test - labels_pred2) / labels_test)) * 100
                content = f'{name} MAPE KNN({i}): {mape2}'
                print(content)
                f.write(content + '\n')
                dump(model2, f"model_{name}.joblib")


        # # 平均值模型
        # mape3 = np.mean(np.abs((labels_test - np.mean(labels)) / labels_test)) * 100
        # content = f'{name} MAPE AVG: {mape3}'
        # print(content)
        # f.write(content + '\n')

        # # 多平均数模型
        # mape4 = 0
        # features_lists = features_test.tolist()
        # for i, feature in enumerate(features_lists):
        #     for target in input_list:
        #         if feature == target:
        #             mape4 += abs((labels_test[i]/1000) - time_list[input_list.index(target)]) / (labels_test[i]/1000)
        # mape4 /= len(labels_test)
        # mape4 *= 100
        # content = f'{name} MAPE MUL_AVG: {mape4}'
        # print(content)
        # f.write(content + '\n')
            
        # model = AdaBoostRegressor()
        # model.fit(features_train, labels_train)
        # labels_pred = model.predict(features_test)
        # mape = np.mean(np.abs((labels_test - labels_pred) / labels_test)) * 100
        # content = f'{name} MAPE GBR: {mape}'
        # print(content)
        # f.write(content + '\n')
