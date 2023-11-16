import pandas as pd
import torch
import torch.nn as nn
# from omnixai.data.timeseries import Timeseries
# from omnixai.explainers.timeseries import MACEExplainer, ShapTimeseries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from explainers import do_WindowSHAP, do_comte
from models import V1Classifier
from train import train
from util import heat_map

N_FEATS = 1
CIDS = [-1]



# #pd_x, pd_y, t_data_x, t_data_y = gen_multivar_regression_casual_data(num_features=N_FEATS, causal_y_idxs=CIDS)
#
# train_pd_x, train_pd_y, train_tr_data_x, train_tr_data_y, _, _ = gen_multivar_classification_casual_data(num_fake_samples=10000, num_features=1, num_classes=2)
# test_pd_x, test_pd_y, test_tr_data_x, test_tr_data_y, _, _ = gen_multivar_classification_casual_data(num_fake_samples=200, num_features=1, num_classes=2)
#
# save_data(train_pd_x, "data/train_pd_x.pt")
# save_data(train_pd_y, "data/train_pd_y.pt")
# save_data(test_pd_x, "data/test_pd_x.pt")
# save_data(test_pd_y, "data/test_pd_y.pt")
#
# train_pd_x = load_data("data/train_pd_x.pt")
# train_pd_y = load_data("data/train_pd_y.pt")
# test_pd_x = load_data("data/test_pd_x.pt")
# test_pd_y = load_data("data/test_pd_y.pt")

train_data = pd.read_csv("data/ECG200_TRAIN.txt", delimiter=r"\s+")
train_pd_x = train_data.iloc[:,1:]
train_pd_y = train_data.iloc[:,0]
train_pd_y = (train_pd_y-train_pd_y.min())/(train_pd_y.max()-train_pd_y.min())

test_data = pd.read_csv("data/ECG200_TEST.txt", delimiter=r"\s+")
test_pd_x = test_data.iloc[:,1:]
test_pd_y = test_data.iloc[:,0]
test_pd_y = (test_pd_y-test_pd_y.min())/(test_pd_y.max()-test_pd_y.min())

###################################### Simple Models ######################################################

model1_LinReg = LinearRegression()
model2_RndFrst = RandomForestClassifier(n_estimators=100)

#data_size = np.prod(list(t_data_x[0].shape))




###################################### Training ######################################################

#model1_LinReg.fit(pd_x, pd_y)
#model2_RndFrst.fit(pd_x, pd_y)

#model = LinearRegression()
#model = RandomForestClassifier(n_estimators=100)
model = V1Classifier(n_feats=96, n_classes=2)


model = train(model, train_pd_x, train_pd_y, n_epochs=2, lr=0.01, loss_fn=nn.CrossEntropyLoss())

###################################### Explanations ######################################################

#do_WindowSHAP(model, train_pd_x, test_pd_x)

do_comte(model, test_pd_x, test_pd_y)