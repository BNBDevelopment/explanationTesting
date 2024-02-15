import os
from pathlib import Path

import math
import pickle

import pandas as pd
import torch
import numpy as np
import re

Y_GEN_FUNCTIONS = {
    "simple":  lambda list_vals : (sum(list_vals) / 1.5) ** 1.2
}


def save_data(data, filepath):
    file = open(filepath, 'wb')
    pickle.dump(data, file)
    file.close()


def load_data(filepath):
    file = open(filepath, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def gen_multivar_classification_casual_data(num_fake_samples=1000, num_features=4, num_classes=2, math_fn=Y_GEN_FUNCTIONS['simple']):
    fake_x = torch.rand([num_fake_samples, num_features])
    data_y = []
    data_x = fake_x

    #np_vect_fn = np.vectorize(math_fn)
    #temp_data_x = np_vect_fn(fake_x)
    temp_data_x = []
    for x in fake_x:
        temp_data_x.append(math_fn(x))
    temp_data_x = torch.FloatTensor(temp_data_x)

    class_end_vals = [i/num_classes for i in range(1,num_classes+1)]
    temp_data_x_norm = (temp_data_x - torch.min(temp_data_x)) / (torch.max(temp_data_x) - torch.min(temp_data_x))

    for item in temp_data_x_norm:
        label = 0
        for class_idx, top_of_range in enumerate(class_end_vals):
            if item > top_of_range:
                label = class_idx+1
        data_y.append(label)

    data_y = torch.FloatTensor(data_y)

    pd_x = pd.DataFrame(data_x.numpy())
    pd_y = pd.DataFrame(data_y.numpy())

    print(f"binary label ratio: {sum(data_y)/num_fake_samples}")
    return pd_x, pd_y, data_x, data_y, num_classes, num_features



def gen_multivar_regression_casual_data(num_fake_samples=1000, window_size=1, window_offset=0, num_features=4, causal_y_idxs=[-2, -1], causal_y_window_items=[0], math_fn=Y_GEN_FUNCTIONS['simple']):
    ######################################## Construct fake causal data ###################################################
    # window_size = 5
    # window_offset = 2

    fake_x = torch.rand([num_fake_samples+(window_size-window_offset+1), num_features])
    data_y = []
    data_x = []

    for i in range(window_offset, num_fake_samples+window_offset):
        window_items = [fake_x[j].unsqueeze(dim=0) for j in range(i-window_offset, i+window_size-window_offset)]
        window = torch.concat(window_items, dim=0)
        data_x.append(window)

        result_y = []
        for window_idx in causal_y_window_items:
            temp = window[window_idx]
            for causal_val in causal_y_idxs:
                result_y.append(temp[causal_val].detach())
        data_y.append(torch.tensor(math_fn(result_y)))

    t_data_x = torch.stack(data_x, dim=0)
    t_data_y = torch.stack(data_y, dim=0)

    print(f"t_data_x: {t_data_x.shape}")
    print(f"t_data_y: {t_data_y.shape}")

    ######################################## Data as pandas df ###################################################
    assert t_data_x.shape[1] == 1, "These methods only seem to work with 2D data so far"
    t_data_x = t_data_x.squeeze(1)
    cols = ["x"+str(i_lbl) for i_lbl in range(1, num_features+1)]

    pd_x = pd.DataFrame(t_data_x.numpy(), columns=cols)
    pd_y = pd.DataFrame(t_data_y.numpy(), columns=["y"])

    print(f"pd_x: {pd_x}")
    return pd_x, pd_y, t_data_x, t_data_y


def filter_data_check(train_data, cutoff_seq_len):
    passes = True

    # if train_data.shape[0] < cutoff_seq_len:
    #     passes = False
    #
    # if train_data['Hours'].iloc[-1] < 47.0:
    #     passes = False

    return passes


glasgow_fails = 0
def convert_glascow(row_item):
    try:
        return float(re.sub("[^0-9]", "", str(row_item)))
    except:
        #print("Malformed Categorical Row")
        return 0.0


def data_preproc(train_data, categorical_feats, cutoff_seq_len):
    #print("Status - Starting Data PreProcessing")
    # CLEANING DATA
    # replace all NaN values with 0
    # train_data[train_data != train_data] = 0
    # train_data = train_data.drop(labels=categorical_feats, axis=1)
    for cat in categorical_feats:
        cat_col = train_data[cat]
        cat_col = cat_col.apply(convert_glascow)
        train_data[cat] = cat_col

    clean_row = np.expand_dims(train_data.to_numpy(), axis=0)[:, :cutoff_seq_len, :]
    return clean_row

preinit_feats = {
    "Capillary refill rate": 0.0,
    "Diastolic blood pressure": 59.0,
    "Fraction inspired oxygen": 0.21,
    "Glascow coma scale eye opening": 4,
    "Glascow coma scale motor response": 6,
    "Glascow coma scale total": 15,
    "Glascow coma scale verbal response": 5,
    "Glucose": 128.0,
    "Heart Rate": 86,
    "Height": 170.0,
    "Mean blood pressure": 77.0,
    "Oxygen saturation": 98.0,
    "Respiratory rate": 19,
    "Systolic blood pressure": 118.0,
    "Temperature": 36.6,
    "Weight": 81.0,
    "pH": 7.4,
}

def data_postproc(train_x, categorical_idx, inorder_col_list):
    print("Status - Starting Data PostProcessing")
    for r, row in enumerate(train_x):
        last_feature_vect = [preinit_feats[x] for x in inorder_col_list if x in preinit_feats.keys()]

        for j, timepoint in enumerate(row):
            #carry forward values
            for i, feat in enumerate(timepoint):
                if feat != feat: #is nan
                    timepoint[i] = last_feature_vect[i]
                else:
                    last_feature_vect[i] = timepoint[i]
                # TODO: exclude categoricals from mean std update
                # if i in categorical_idx:
                #     timepoint[i] = (timepoint[i] - feat_mean[i]) / feat_std[i]
            row[j] = timepoint
        train_x[r] = row


    if np.isnan(train_x).any():
        raise ValueError("Failure, found at least one NaN in the training data after carry-forward was implemented")
    masked_mean = np.nanmean(train_x, axis=0)
    masked_std = np.nanstd(train_x, axis=0)
    for cat in categorical_idx:
        masked_mean[:,cat] = 0
        masked_std[:, cat] = 1

    train_x = (train_x - masked_mean) / masked_std
    return train_x



            #clean_row = normalize_vals(clean_row)

    #return clean_row


def load_mimic_binary_classification(base_path, filename, datatype, cutoff_seq_len=30, num_features=18, categorical_feats=[], excludes=[]):
    print("Status - Loading MIMIC-III data")
    val_file_name = "val_listfile.csv"
    test_file_name = "test_listfile.csv"

    train_file = pd.read_csv(base_path / filename)
    #train_y = train_file["y_true"]
    train_stay_ref = train_file["stay"]

    notused_seq_lens = []
    used_seq_lens = []

    read_folder = datatype
    if datatype == "val":
        read_folder = "train"

    clean_data = []
    matching_ys = []

    for i, stay_ref in enumerate(train_stay_ref):
        train_data = pd.read_csv(base_path / (read_folder+"/"+stay_ref))
        for drop_col in excludes:
            train_data = train_data.drop(drop_col, axis=1)

        passes_filter = filter_data_check(train_data, cutoff_seq_len)

        if passes_filter:

            clean_row = data_preproc(train_data, categorical_feats, cutoff_seq_len)
            clean_data.append(clean_row)
            matching_ys.append(train_file["y_true"].iloc[i])

            used_seq_lens.append(clean_row.shape[1])
        else:
            notused_seq_lens.append(train_data.shape[0])

    max_used_len = max(used_seq_lens)
    padded_items = []
    for item in clean_data:
        padlen = max_used_len-item.shape[1]
        padded_items.append(np.pad(item, ((0,0), (0,padlen), (0,0))))

    train_x = np.zeros((0, max_used_len, num_features))
    train_y = np.zeros((0))
    train_x = np.concatenate((padded_items), axis=0)
    train_y = np.expand_dims(np.stack((matching_ys), axis=0), axis=1)

    categorical_idx = [i for i in range(len(train_data.columns)) if train_data.columns[i] in categorical_feats]
    train_x = data_postproc(train_x, categorical_idx, list(train_data.columns))

    return train_x, train_y


def load_file_data(config, data_type="train"):
    datadir = config['datadir']
    mimic_data_path = config['mimic_data_path']
    categoricals = config['categorical_features']
    exclude_list  = config['excludes']
    max_seq_len = config['max_seq_len']
    num_features = config['num_features']
    data_filename = config['data_filename']

    full_path = datadir + data_type+data_filename
    x_path = full_path + "_x.csv"
    y_path = full_path + "_y.csv"
    if os.path.isfile(x_path) and os.path.isfile(y_path) and config['load_data']:
        train_x = load_data(x_path)
        train_y = load_data(y_path)
    else:
        path = Path(mimic_data_path)

        train_x, train_y = load_mimic_binary_classification(path, data_type+"_listfile.csv", data_type, cutoff_seq_len=max_seq_len,
                                                            num_features=num_features, categorical_feats=categoricals, excludes=exclude_list)
        save_data(train_x, x_path)
        save_data(train_y, y_path)

    return train_x, train_y