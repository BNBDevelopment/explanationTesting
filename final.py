import os.path
import pickle
from pathlib import Path

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from omnixai.data.timeseries import Timeseries
# from omnixai.explainers.timeseries import MACEExplainer, ShapTimeseries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

import dataset
import gru_ode_bayes
import torchmimic.models
from TLSTM import TLSTM
from explanation_methods import do_WindowSHAP, do_comte
from mimic3models.in_hospital_mortality import utils
from mimic3models.preprocessing import Normalizer, Discretizer
from models import V1Classifier, BasicLSTM, select_model
from train import train#, bayes_train
import tensorflow
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
#model = V1Classifier(n_feats=96, n_classes=2)


#model = train(model, train_pd_x, train_pd_y, n_epochs=20, lr=0.01, loss_fn=nn.CrossEntropyLoss())

###################################### Explanations ######################################################

#do_WindowSHAP(model, train_pd_x, test_pd_x)

#do_comte(model, train_pd_x, train_pd_y, test_pd_x, test_pd_y)






def main():
    cutoff_seq_len = 50
    num_features = 18
    excludes = ['Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale verbal response']

    path = "./data/in-hospital-mortality/"

    train_x, train_y = load_file_data(datadir="./data/my_mimic/", mimic_data_path=path, exclude_feats_list=excludes, data_type="train")

    val_x, val_y = load_file_data(datadir="./data/my_mimic/", mimic_data_path=path, exclude_feats_list=excludes, data_type="val")

    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model_type = "benchmark_lstm"
    #model_type = ""
    #model_type = "xgboost"
    model_type = "benchmark_lstm"

    wm = True
    mt = 'lstm'

    model = select_model(model_type)
    model = train(model, basic_config, train_x, train_y, val_x, val_y)

    #test_x, test_y = load_mimic_binary_classification(path, "test_listfile.csv", "test", cutoff_seq_len=cutoff_seq_len, num_features=num_features, categorical_feats=excludes)
    test_x, test_y = load_file_data(datadir="./data/my_mimic/", mimic_data_path=path, exclude_feats_list=excludes,
                                  data_type="test")

    ft = ["Hours","Capillary refill rate","Diastolic blood pressure","Fraction inspired oxygen",
          "Glascow coma scale total","Glucose","Heart Rate","Height","Mean blood pressure","Oxygen saturation",
          "Respiratory rate","Systolic blood pressure","Temperature","Weight","pH"]


    #model = torch.load("model_epoch15_aucroc0.6252150479891426.pt")
    do_WindowSHAP(model, train_x, test_x, feature_names=ft, wrap_model=wm, model_type=mt)
    #do_comte(model, train_x, train_y, test_x, test_y, test_id_to_explain=0)


if __name__ == "__main__":
    main()