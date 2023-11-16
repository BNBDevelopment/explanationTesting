#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset

from dataset import TimeseriesDataset
from model import LSTMClassification

import timeshap
from timeshap.wrappers import TorchModelWrapper
from timeshap.utils import calc_avg_event
from timeshap.utils import get_avg_score_with_avg_event
from timeshap.explainer import local_report

import sklearn
from sklearn.linear_model import LinearRegression


# In[44]:


def make_dataloaders(data_x, data_y, window_size, batch_size, window_offset):
    train_set = TimeseriesDataset(
        torch.from_numpy(data_x.to_numpy()),
        torch.from_numpy(data_y.to_numpy()),
        window_size=window_size,
        window_offset=window_offset)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
    )

    n_features = train_set.tensors[0].shape[1]
    example_shape = train_set.tensors[0][0].shape

    return train_set, train_loader, n_features, [1, window_size, n_features]


# In[54]:


def train(model, train_loader, optimizer, epoch, log_interval, losses, device, loss_fn=nn.functional.mse_loss):
    model.train()
    epoch_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device).to(torch.float32)
        target = target.to(device).to(torch.float32)
        output = model(data)

        loss = loss_fn(output.squeeze(), target.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        epoch_loss.append(loss.item())

    print(f"Train Epoch: {epoch} \t\t Epoch Loss: {sum(epoch_loss)/len(epoch_loss):.6f}")


# In[117]:


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.start_flatten = nn.Flatten()
        self.linear = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.start_flatten(x)
        pred = self.linear(x)
        return pred


# In[4]:


data = pd.read_csv("../../multivariate-attention-tcn/Datasets/AirQualityUCI.csv", sep=';', decimal=',')
data = data.drop('Date', axis=1)
data = data.drop('Time', axis=1)
data.infer_objects()
drops = [-1*x for x in list(range(0,12))]
data = data.drop(data.columns[drops], axis=1)
data = data.dropna(axis=0)


# In[135]:


train_size = 0.8
val_size = 0.1
test_size = 0.1
train_df = data.iloc[:round(train_size*data.shape[0])]
val_df = data.iloc[round(train_size*data.shape[0]):round(train_size*data.shape[0]+val_size*data.shape[0])]
test_df = data.iloc[round(train_size*data.shape[0]+val_size*data.shape[0]):round(train_size*data.shape[0]+val_size*data.shape[0]+test_size*data.shape[0])]

print(f"train_df.shape: {train_df.shape}")
train_dy = train_df.iloc[:, -1:]
train_dx = train_df.iloc[:,:-1]
val_dy = val_df.iloc[:, -1:]
val_dx = val_df.iloc[:,:-1]
test_dy = test_df.iloc[:, -1:]
test_dx = test_df.iloc[:,:-1]

print(f"train_dy.shape: {train_dy.shape}")


# In[110]:


current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(current_device)

window_size = 5
window_offset = -2
# window_size = 1
# window_offset = 0
batch_size = 32

#model params
target_size = 1
n_layer = 3

#training params
lr = 0.01
epochs = 1


# In[111]:


train_ds, train_loader, n_features, example_shape = make_dataloaders(train_dx, train_dy, window_size, batch_size, window_offset)
print(f"example_shape: {example_shape}")


# In[118]:


model = LinearRegressionModel(input_size=np.prod(example_shape))


# In[119]:


if __name__ == "__main__":

    if issubclass(model.__class__, nn.Module):
        model.to(current_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []
        for i in range(1, epochs+1):
            train(model, train_loader, optimizer, epoch=i, log_interval=50, losses=losses, device=current_device)
    else:
        model.fit(train_dx, train_dy)
        score = model.score(val_dx, val_dy)
        print(f"score: {score}")


# In[103]:


def convertDataloaderToPd(dataloader):
    temp_x = None
    temp_y = None
    for x in dataloader:
        #print(f"temp_x: {temp_x}")
        if temp_x is None:
            temp_x = x[0].detach().cpu().numpy()
        else:
            np.concatenate([temp_x, x[0].detach().cpu().numpy()], axis=0)

        if temp_y is None:
            temp_y = x[1].detach().cpu().numpy()
        else:
            np.concatenate([temp_y, x[1].detach().cpu().numpy()], axis=0)

    #return pd.DataFrame(temp_x), pd.DataFrame(temp_y)
    return pd.DataFrame(temp_x.reshape(-1, temp_x.shape[-1])), pd.DataFrame(temp_y.reshape(-1, temp_y.shape[-1]))


# In[114]:


test_ds, test_loader, n_features, example_shape = make_dataloaders(test_dx, test_dy, window_size, batch_size, window_offset)


# In[134]:


test_dx


# In[139]:


from timeshap.utils import calc_avg_sequence
average_sequence = test_dx.mean(axis=0)


# In[156]:


average_sequence = pd.DataFrame(average_sequence).transpose()


# In[126]:


import timeshap


# In[149]:


model_wrapped = TorchModelWrapper(model)
f_hs = lambda x, y=None: model_wrapped.predict_last_hs(x, y)


# In[147]:


model_features = list(test_dx.columns)
plot_feats = {k:k for k in list(test_dx.columns)}


# In[159]:


from timeshap.explainer import local_report
test_df['holder'] = 'y'
pruning_dict = {'tol': 0.025}
event_dict = {'rs': 42, 'nsamples': 32000}
feature_dict = {'rs': 42, 'nsamples': 32000, 'feature_names': model_features, 'plot_features': plot_feats}
cell_dict = None
local_report(f_hs, test_df, pruning_dict, event_dict, feature_dict, cell_dict, average_sequence, entity_col='holder', model_features=model_features)

