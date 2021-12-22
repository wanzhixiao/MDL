import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import *
import json

class MinMaxScalar:
    def __init__(self, min, max):
        self._min = min
        self._max = max

    def __int__(self):
        pass

    def transform(self, data):
        return (data - self._min) / (self._max - self._min)

    def fit_transform(self, data):
        self._min = np.min(data)
        self._max = np.max(data)
        return (data - self._min) / (self._max - self._min)

    def inverse_transform(self, data):
        return (data * (self._max - self._min)) + self._min


class StandardScaler:
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

    def __int__(self):
        pass

    def transform(self, data):
        return (data - self._mean) / self._std

    def fit_transform(self, data):
        self._mean = np.mean(data)
        self._std = np.std(data)
        return (data - self._mean) / self._std

    def inverse_transform(self, data):
        return (data * self._std) + self._mean

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class MDLDataSet(Dataset):
    def __init__(self, flow_data, od_flow_data, flow_label_data, od_flow_label_data):
        '''
        :param flow: (n_samples,hist_len,channel,height,width)
        :param od_flow: (n_samples,hist_len,channel,height,width)
        :param flow_label: (n_samples,1,channel,height,width)
        :param od_flow_label: (n_samples,1,channel,height,width)
        '''
        self.flow_data = torch.from_numpy(flow_data).type(torch.FloatTensor)
        self.od_flow_data = torch.from_numpy(od_flow_data).type(torch.FloatTensor)
        self.flow_label_data = torch.from_numpy(flow_label_data).type(torch.FloatTensor)
        self.od_flow_label_data = torch.from_numpy(od_flow_label_data).type(torch.FloatTensor)

    def __len__(self):
        """
        :return: length of dataset (number of samples).
        """
        return self.flow_data.shape[0]

    def __getitem__(self, index):
        """
        :param index: int, range between [0, length - 1].
        :return:
            x: torch.tensor, (B,T,C,H,W)
            y: torch.tensor, (B,1,C,H,W)
        """
        flow = self.flow_data[index]
        od_flow = self.od_flow_data[index]
        flow_label = self.flow_label_data[index]
        od_flow_label = self.od_flow_label_data[index]

        return {"flow": flow, "od_flow": od_flow, "flow_label": flow_label, "od_flow_label": od_flow_label}

def create_seq(len_trend, len_period, len_closeness):
    # inout_flow = np.random.rand(744, 2, 16, 16)
    # od_flow = np.random.rand(744, 512, 16, 16)

    inout_flow = np.load(in_out_file)['data']
    od_flow = np.load(od_file)['data']

    print(inout_flow.shape)
    print(od_flow.shape)

    total_time_steps = inout_flow.shape[0]

    start = max(24 * 7 * len_trend, max(24 * len_period, len_closeness))

    flow_data_arr, od_flow_data_arr = [], []

    flow_label_arr, od_flow_label_arr = [], []

    for i in range(start, total_time_steps):
        len1, len2, len3 = len_trend, len_period, len_closeness
        flow_list = []
        od_flow_list = []

        while len1 > 0:
            flow_trend = inout_flow[i - 24 * 7 * len1]  # i-24*7*3, i-24*7*2 i-24*7*1
            od_flow_trend = od_flow[i - 24 * 7 * len1]
            flow_list.append(flow_trend)
            od_flow_list.append(od_flow_trend)
            len1 = len1 - 1

        while len2 > 0:
            flow_peroid = inout_flow[i - 24 * len2]  # i-24*3, i-24*2 i-24*1
            od_flow_peroid = od_flow[i - 24 * len2]
            flow_list.append(flow_peroid)
            od_flow_list.append(od_flow_peroid)
            len2 = len2 - 1

        while len3 > 0:
            flow_closeness = inout_flow[i - len3]  # i-3, i-2 i-1
            od_flow_closeness = od_flow[i - len3]
            flow_list.append(flow_closeness)
            od_flow_list.append(od_flow_closeness)
            len3 = len3 - 1

        flow_label = inout_flow[i:i + 1]
        od_flow_label = od_flow[i:i + 1]

        flow_data_arr.append(flow_list)
        od_flow_data_arr.append(od_flow_list)

        flow_label_arr.append(flow_label)
        od_flow_label_arr.append(od_flow_label)

    flow_data_arr = np.array(flow_data_arr)
    od_flow_data_arr = np.array(od_flow_data_arr)
    flow_label_arr = np.array(flow_label_arr)
    od_flow_label_arr = np.array(od_flow_label_arr)

    # 此处可以保存成npy
    return flow_data_arr, od_flow_data_arr, flow_label_arr, od_flow_label_arr

def get_dataloader(len_trend, len_period, len_closeness, train_prop, val_prop, batch_size):

    def create_loader(flow, od_flow, flow_label, od_flow_label, batch_size, flow_scaler,od_flow_scaler,shuffle=False):

        flow, flow_label = flow_scaler.transform(flow), flow_scaler.transform(flow_label)

        od_flow, od_flow_label = od_flow_scaler.transform(od_flow), od_flow_scaler.transform(od_flow_label)

        dataset = MDLDataSet(flow, od_flow, flow_label, od_flow_label)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    #1.构建数据集
    flow_data, od_flow_data, flow_label, od_flow_label = create_seq(len_trend, len_period, len_closeness)

    #2.划分训练集,验证集,测试集 = 8:1:1
    num_samples = flow_data.shape[0]
    num_train = int(num_samples * train_prop)
    num_val = int(num_samples * val_prop)
    num_test = num_samples - num_train - num_val

    train_flow_data, train_od_flow_data, train_flow_label, train_od_flow_label = \
        flow_data[:num_train], od_flow_data[:num_train], flow_label[:num_train], od_flow_label[:num_train]

    val_flow_data, val_od_flow_data, val_flow_label, val_od_flow_label = \
        flow_data[num_train:num_train + num_val], od_flow_data[num_train:num_train + num_val], \
        flow_label[num_train:num_train + num_val], od_flow_label[num_train:num_train + num_val]

    test_flow_data, test_od_flow_data, test_flow_label, test_od_flow_label = \
        flow_data[-num_test:], od_flow_data[-num_test:], flow_label[-num_test:], od_flow_label[-num_test:]

    flow_scaler = MinMaxScalar(np.min(train_flow_data), np.max(train_flow_data))
    od_flow_scaler = MinMaxScalar(np.min(train_od_flow_data), np.max(train_od_flow_data))

    #3.构建dataloader
    train_loader = create_loader(train_flow_data, train_od_flow_data, train_flow_label, train_od_flow_label,batch_size,\
                               flow_scaler,od_flow_scaler,shuffle=True)
    val_loader = create_loader(val_flow_data, val_od_flow_data, val_flow_label, val_od_flow_label, batch_size, \
                               flow_scaler, od_flow_scaler,shuffle=False)
    test_loader =create_loader(test_flow_data, test_od_flow_data, test_flow_label, test_od_flow_label, batch_size,\
                               flow_scaler,od_flow_scaler,shuffle=False)

    return {
        'train' : train_loader,
        'validate' : val_loader,
        'test' : test_loader,
        'flow_scaler' : flow_scaler,
        'od_flow_scaler' : od_flow_scaler
    }

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels!=null_val)

    mask = mask.astype('float32')
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    #modified
    loss = np.abs((preds-labels)/labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)


def mean_absolute_percentage_error(y_true, y_pred):
    '''
    caculate mape
    :param y_true:
    :param y_pred:
    :return: mape ∈ [0,+ꝏ]
    '''
    return masked_mape(y_true,y_pred)
    # return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def evaluate(flow_pred, flow_targets, od_pred, od_targets, flow_scaler, od_flow_scaler):
    '''
    evaluate model with rmse, mae and mape
    :param flow_pred: [n_samples, 2, hegiht, width]
    :param flow_targets: [n_samples, 2, hegiht, width]
    :param od_pred: [n_samples, 2*height*width, hegiht, width]
    :param od_targets: [n_samples, 2*height*width, hegiht, width]
    :return:
    '''
    metrics = defaultdict(dict)
    #inverse_transform
    flow_pred = flow_scaler.inverse_transform(flow_pred)
    flow_targets = flow_scaler.inverse_transform(flow_targets)
    od_pred = od_flow_scaler.inverse_transform(od_pred)
    od_targets = od_flow_scaler.inverse_transform(od_targets)

    flow_pred = np.reshape(flow_pred, (flow_pred.shape[0], -1))
    flow_targets = np.reshape(flow_targets, (flow_targets.shape[0], -1))

    od_pred = np.reshape(od_pred, (od_pred.shape[0], -1))
    od_targets = np.reshape(od_targets, (od_targets.shape[0], -1))


    flow_rmse = np.sqrt(mean_squared_error(flow_targets,flow_pred))
    flow_mae = mean_absolute_error(flow_targets,flow_pred)
    flow_mape = mean_absolute_percentage_error(flow_targets,flow_pred)

    od_flow_rmse = np.sqrt(mean_squared_error(od_targets,od_pred))
    od_flow_mae = mean_absolute_error(od_targets,od_pred)
    od_flow_mape = mean_absolute_percentage_error(od_targets, od_pred)

    metrics['flow_rmse'] = flow_rmse
    metrics['flow_mae'] = flow_mae
    metrics['flow_mape'] = flow_mape
    metrics['od_flow_rmse'] = od_flow_rmse
    metrics['od_flow_mae'] = od_flow_mae
    metrics['od_flow_mape'] = od_flow_mape

    return metrics

def save_model(path, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict,path)

'''
if __name__ == '__main__':
    len_trend, len_period, len_closeness = 0, 0, 3

    train_loader, val_loader, test_loader, flow_scaler,od_flow_scaler = \
        get_data_loader(len_trend, len_period, len_closeness, train_prop=0.8, val_prop=0.1, batch_size=64)

    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))

    for iter, data in enumerate(tqdm(train_loader)):
        flow = data['flow']
        od_flow = data['od_flow']
        flow_label = data['flow_label']
        od_flow_label = data['od_flow_label']

        print('flow:{},od_flow:{},flow_label:{},od_flow_label:{}'.format(flow.shape,od_flow.shape,\
                                                                         flow_label.shape,od_flow_label.shape))
        print(flow)
        print(od_flow)
        print(flow_label)
        print(od_flow_label)
        break
        '''
