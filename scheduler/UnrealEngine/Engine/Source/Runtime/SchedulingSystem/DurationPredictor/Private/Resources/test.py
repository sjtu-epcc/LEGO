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
        model = load(f'model_{name}.joblib')
        with open(f'output_{name}.txt', 'r') as f2:
            with open(f"duration_{name}.txt", "w") as f3:
                for line in f2:
                    row = list(map(float, line.split(',')))
                    if len(row) == length:
                        data.append(row)
                        if row[:-1] not in input_list:
                            input_list.append(row[:-1])
                            time = model.predict([row[:-1]])
                            dim = row[:-1]
                            for i in range(len(dim)):
                                f3.write(f"{int(dim[i])},")
                            f3.write(f"{time[0]}\n")
            print(f"duration_{name}.txt has been created.")
