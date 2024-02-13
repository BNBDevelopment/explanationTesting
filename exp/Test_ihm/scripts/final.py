import os.path
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
from models import V1Classifier
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


#loading MIMIC data:

def load_mimic_binary_classification(base_path, filename, datatype, cutoff_seq_len=30, num_features=18, categorical_feats=[]):

    val_file_name = "val_listfile.csv"
    test_file_name = "test_listfile.csv"

    train_file = pd.read_csv(base_path / filename)
    #train_y = train_file["y_true"]
    train_stay_ref = train_file["stay"]

    total_data_records = 0
    useable_records = 0

    train_x = np.zeros((0, cutoff_seq_len, num_features-len(categorical_feats)))
    train_y = np.zeros((0))

    seq_lens = []

    read_folder = datatype
    if datatype == "val":
        read_folder = "train"

    for i, stay_ref in enumerate(train_stay_ref):
        train_data = pd.read_csv(base_path / (read_folder+"/"+stay_ref))

        total_data_records += 1

        seq_lens.append(train_data.shape[0])
        if train_data.shape[0] >= cutoff_seq_len:
            useable_records += 1

            #replace all NaN values with 0
            train_data[train_data != train_data] = 0
            #TODO: replace - drop categoricals, needed since previous user did not write data correctly (not consistent categortical values, poorly formatted, etc...)
            train_data = train_data.drop(labels=categorical_feats, axis=1)

            clean_data = np.expand_dims(train_data.to_numpy(), axis=0)[:, :cutoff_seq_len, :]
            train_x = np.concatenate((train_x, clean_data), axis=0)

            train_y = np.concatenate((train_y, np.expand_dims(train_file["y_true"].iloc[i], axis=0)), axis=0)


    return train_x, train_y


def load_file_data(datadir="./", mimic_data_path="./data/in-hospital-mortality/", max_seq_len=70, num_features=18,
                   exclude_feats_list=None, data_type="train"):

    if os.path.isfile(datadir + data_type+"_ihm_30_x.csv") and os.path.isfile(datadir + data_type+"_ihm_30_y.csv"):
        train_x = dataset.load_data(datadir + data_type+"_ihm_30_x.csv")
        train_y = dataset.load_data(datadir + data_type+"_ihm_30_y.csv")
    else:
        path = Path(mimic_data_path)

        train_x, train_y = load_mimic_binary_classification(path, data_type+"_listfile.csv", data_type, cutoff_seq_len=max_seq_len,
                                                            num_features=num_features, categorical_feats=exclude_feats_list)
        dataset.save_data(train_x, datadir + data_type+"_ihm_30_x.csv")
        dataset.save_data(train_y, datadir + data_type+"_ihm_30_y.csv")

    return train_x, train_y

def main():
    cutoff_seq_len = 50
    num_features = 18
    excludes = ['Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale verbal response']

    path = "./data/in-hospital-mortality/"

    train_x, train_y = load_file_data(datadir="./data/my_mimic/", mimic_data_path=path, exclude_feats_list=excludes, data_type="train")

    val_x, val_y = load_file_data(datadir="./data/my_mimic/", mimic_data_path=path, exclude_feats_list=excludes, data_type="val")

    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = "torchmimic_lstm"




    ### Basic Test Model #############################################################################################
    if model_type == "basic":
        # for imbalanced dataset:
        weights = [1 / (1 - (sum(train_y) / len(train_y))), 1 / (sum(train_y) / len(train_y))]
        class_weights = torch.FloatTensor(weights).cuda()
        balancedCE = nn.CrossEntropyLoss(weight=class_weights)
        #model params
        lr = 0.001
        model = V1Classifier(n_feats=(num_features - len(excludes)) * cutoff_seq_len, n_classes=2)
        basic_config = {
            'n_epochs': 10,
            'lr': lr,
            'batch_size': 32,
            'optimizer': torch.optim.Adam(model.parameters(), lr=lr),
            'device': current_device,
            'loss_fn': balancedCE,
        }
        model = train(model, basic_config, train_x, train_y)


    ### GRU ODE Bayes Model ##########################################################################################
    elif model_type == "gru_ode_bayes":
        gob_model_config = {
            'input_size': num_features,
            'hidden_size': 50,
            'p_hidden': 25,
            'prep_hidden': 10,
            'logvar': True,
            'mixing': 1e-4,
            'delta_t': 0.1,
            'T': 200,
            'lambda': 0,
            'classification_hidden': 2,
            'cov_size': cutoff_seq_len,
            'cov_hidden': 50,
            'dropout_rate': 0.2,
            'full_gru_ode': True,
            'no_cov': True,
            'impute': False,
            'lr': 0.001,
            'weight_decay': 0.0001,
        }
        model = gru_ode_bayes.NNFOwithBayesianJumps(input_size=gob_model_config["input_size"],
                                                      hidden_size=gob_model_config["hidden_size"],
                                                      p_hidden=gob_model_config["p_hidden"],
                                                      prep_hidden=gob_model_config["prep_hidden"],
                                                      logvar=gob_model_config["logvar"],
                                                      mixing=gob_model_config["mixing"],
                                                      classification_hidden=gob_model_config["classification_hidden"],
                                                      cov_size=gob_model_config["cov_size"],
                                                      cov_hidden=gob_model_config["cov_hidden"],
                                                      dropout_rate=gob_model_config["dropout_rate"],
                                                      full_gru_ode=gob_model_config["full_gru_ode"],
                                                      impute=gob_model_config["impute"])
        lr = 0.001
        weight_decay = 0.0001
        gob_train_config = {
            'n_epochs': 10,
            'lr': lr,
            'batch_size': 32,
            'optimizer': torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
            'device': current_device,
            'loss_fn': torch.nn.BCEWithLogitsLoss(reduction='sum'),
        }
        model.to(gob_train_config['device'])
        model = bayes_train(model, gob_train_config, train_x, train_y)


    ### T-LSTM Model ##########################################################################################
    elif model_type == "tlstm":
        lr = 0.001
        basic_config = {
            'n_epochs': 10,
            'lr': lr,
            'batch_size': 32,
            'device': current_device,
            'loss_fn': balancedCE,
        }

        model = TLSTM(input_dim=15, output_dim=2, hidden_dim=512, fc_dim=64,key=1)
        cross_entropy, y_pred, y, logits, labels = model.get_cost_acc()

        basic_config['optimizer'] = tensorflow.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)
        model = tf_train(model, basic_config, train_x, train_y)


    elif model_type == "torchmimic_lstm":
        from torchmimic.benchmarks import IHMBenchmark

        root_dir = "data/in-hospital-mortality"

        from torchmimic.models import StandardLSTM

        model = StandardLSTM(
            n_classes=2,
            hidden_dim=16,
            num_layers=2,
            dropout_rate=0.3,
            bidirectional=False,
        )

        trainer = IHMBenchmark(
            model=model,
            train_batch_size=8,
            test_batch_size=256,
            data=root_dir,
            learning_rate=0.001,
            weight_decay=0,
            report_freq=200,
            device=0,
            sample_size=None,
            wandb=False,
        )

        trainer.fit(100)

    ### Benchmark LSTM Model ##########################################################################################
    elif model_type == "benchmark_lstm":

        # train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
        #                                          listfile=os.path.join(args.data, 'train_listfile.csv'),
        #                                          period_length=48.0)
        #
        # val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
        #                                        listfile=os.path.join(args.data, 'val_listfile.csv'),
        #                                        period_length=48.0)
        # discretizer = Discretizer(timestep=float(args.timestep),
        #                           store_masks=True,
        #                           impute_strategy='previous',
        #                           start_time='zero')
        # discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
        # cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
        # normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
        # normalizer_state = args.normalizer_state
        # if normalizer_state is None:
        #     normalizer_state = 'ihm_ts{}.input_str-{}.start_time-zero.normalizer'.format(args.timestep, args.imputation)
        #     normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        # normalizer.load_params(normalizer_state)
        # train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)
        # val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)

        #model params
        lr = 0.01 #from paper
        dropout = 0.3 #their code says is best

        model = torchmimic.models.StandardLSTM(
            n_classes=2,
            hidden_dim=32, #their code says is best
            num_layers=2, #their code says is best
            dropout_rate=dropout,
            bidirectional=True,
            input_size=15
        )

        basic_config = {
            'n_epochs': 100, #default from their code
            'lr': lr,
            'batch_size': 8, #their code says is best
            'optimizer': torch.optim.Adam(model.parameters(), lr=lr), #from paper
            'device': current_device,
            'loss_fn': nn.BCELoss(), #default from their code
        }
        model = train(model, basic_config, train_x, train_y, val_x, val_y)
    else:
        raise Exception(f"Wrong model type: {model_type}")


    test_x, test_y = load_mimic_binary_classification(path, "test_listfile.csv", "test", cutoff_seq_len=cutoff_seq_len, num_features=num_features, categorical_feats=cts)


    ft = ["Hours","Capillary refill rate","Diastolic blood pressure","Fraction inspired oxygen",
          "Glascow coma scale total","Glucose","Heart Rate","Height","Mean blood pressure","Oxygen saturation",
          "Respiratory rate","Systolic blood pressure","Temperature","Weight","pH"]

    do_WindowSHAP(model, train_x, test_x, feature_names=ft)


if __name__ == "__main__":
    main()