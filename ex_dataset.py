import os
from pathlib import Path

import math
import pickle

import pandas as pd
import torch
import numpy as np
import re

from mimic3models.preprocessing import Normalizer, Discretizer

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

def data_postproc(train_x, categorical_idx, inorder_col_list, config):
    print("Status - Starting Data PostProcessing")
    #TODO: confirm cateogrical is exlcude from eman std update
    #TODO: concat with mask
    orig_or_imputed_mask = (train_x != train_x).astype(float)
    for r, row in enumerate(train_x):
        last_feature_vect = [preinit_feats[x] for x in inorder_col_list if x in preinit_feats.keys()]
        for j, timepoint in enumerate(row):
            #carry forward values
            for i, feat in enumerate(timepoint):
                if feat != feat: #is nan
                    timepoint[i] = last_feature_vect[i]
                else:
                    last_feature_vect[i] = timepoint[i]
            row[j] = timepoint
        train_x[r] = row

    train_x = train_x.astype(float)
    if np.isnan(train_x).any():
        raise ValueError("Failure, found at least one NaN in the training data after carry-forward was implemented")
    masked_mean = np.mean(train_x, axis=0)
    masked_std = np.std(train_x, axis=0)
    for cat in categorical_idx:
        masked_mean[:, cat] = 0
        masked_std[:, cat] = 1

    for location in np.argwhere(masked_std==0):
        masked_std[location] = 1

    train_x = (train_x - masked_mean) / masked_std
    if np.isnan(train_x).any():
        raise ValueError("Failure, found at least one NaN in the training data after mean std normalization")

    if config['input_concat_w_mask']:
        return np.concatenate((train_x, orig_or_imputed_mask), axis=-1)
    else:
        return train_x



            #clean_row = normalize_vals(clean_row)

    #return clean_row


def merge_time_into_windows(train_x, window_size, ts_size):
    hour_index = 0
    final_train = []
    for icu_stay in train_x:
        #icu_stay = icu_stay[0]
        #tps_to_combine = []
        merged_stay = []
        #stay_max_time = np.nanmax(icu_stay[:, hour_index])
        icu_stay = icu_stay.to_numpy()

        windows = [(x*window_size,(x+1)*window_size) for x in range(0,int(ts_size))]
        for window in windows:
            matching_timepoint_idxs = np.argwhere(np.logical_and(icu_stay[:,hour_index] >= window[0], icu_stay[:,hour_index] < window[1]))
            matching_timepoints = icu_stay[matching_timepoint_idxs,:].squeeze()
            if len(matching_timepoints.shape) > 1:
                matching_timepoints = list(matching_timepoints)
            else:
                matching_timepoints = [matching_timepoints]

            if len(matching_timepoints) > 0:
                matching_timepoints.reverse()
                merged_window = matching_timepoints[0]
                for i in range(1, len(matching_timepoints)):
                    replace_idxs = np.argwhere(merged_window != merged_window)
                    if math.prod(replace_idxs.shape) != 0:
                        merged_window[replace_idxs] = matching_timepoints[i][replace_idxs]
                merged_stay.append(merged_window)
            else:
                merged_stay.append(np.ones(icu_stay.shape[-1])*np.nan)

        final_train.append(np.stack(merged_stay[:int(ts_size)]))
    return final_train



def load_mimic_binary_classification(config, base_path, filename, datatype, cutoff_seq_len=30, num_features=18, categorical_feats=[], excludes=[]):
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
    print(f"Using '{config['data_preproc']}' as preprocessing method.")

    for i, stay_ref in enumerate(train_stay_ref):
        train_data = pd.read_csv(base_path / (read_folder+"/"+stay_ref))

        matching_ys.append(train_file["y_true"].iloc[i])

        if config['data_preproc'] == 'PaperDescription':
            passes_filter = filter_data_check(train_data, cutoff_seq_len)
            if passes_filter:
                clean_row = data_preproc(train_data, categorical_feats, cutoff_seq_len)
                clean_data.append(clean_row)
                used_seq_lens.append(clean_row.shape[1])
            else:
                notused_seq_lens.append(train_data.shape[0])
        elif config['data_preproc'] == 'mimic3benchmark':
            temp = train_data.fillna('')
            temp['Glascow coma scale total'] = temp['Glascow coma scale total'].astype(str).apply(
                lambda x: x.split('.')[0])
            temp = temp.to_numpy()
            clean_data.append(temp)
            used_seq_lens.append(temp.shape[1])
        else:
            cat_map = {
                'Glascow coma scale eye opening': {'Spontaneously':4, 'To Pressure':2, 'To Sound':3, 'None':1},
                'Glascow coma scale motor response': {'Obeys Commands':6, 'Localizing':5, 'Normal Flexion':4, 'Abnormal Flexion':3, 'Extension':2, 'None':1},
                #'Glascow coma scale total': {},
                'Glascow coma scale verbal response': {'Oriented':5, 'Confused':4, 'Words':3, 'Sounds':2, 'None':1},
            }

            for cat in categorical_feats:
                train_data[cat] = train_data[cat].apply(lambda x: cat_map[cat][x] if x in cat_map[cat].keys()  else x).apply(lambda y: re.sub("\D", "", y) if type(y) == str else y)
            train_data = train_data.replace("", np.nan)
            clean_data.append(train_data)
            used_seq_lens.append(train_data.shape[1])

    if config['data_preproc'] == 'mimic3benchmark':
        discretizer = Discretizer(timestep=float(config['merge_time_size']),
                                  store_masks=True,
                                  impute_strategy='previous',
                                  start_time='zero')
        #for x in categorical_feats:
        # temp['Glascow coma scale eye opening'] = temp['Glascow coma scale eye opening'].fillna('None')
        # temp['Glascow coma scale motor response'] = temp['Glascow coma scale motor response'].fillna('None')
        # temp['Glascow coma scale verbal response'] = temp['Glascow coma scale verbal response'].fillna('None')

        discretizer_header = discretizer.transform(temp)[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
        normalizer = Normalizer(fields=cont_channels)
        normalizer_state = 'ihm_ts1.0.input_str_previous.start_time_zero.normalizer'
        normalizer.load_params(normalizer_state)

        ts = [config['hours_to_eval'] for i in range(len(clean_data))]
        clean_data = [discretizer.transform(X.astype(str), end=t)[0] for (X, t) in zip(clean_data, ts)]
        if normalizer is not None:
            clean_data = [normalizer.transform(X) for X in clean_data]
        train_x = np.stack(clean_data, axis=0)
        train_y = np.expand_dims(np.stack((matching_ys), axis=0), axis=1)
    else:
        config['ts_size'] = config['hours_to_eval'] // config['merge_time_size']
        clean_data = merge_time_into_windows(clean_data, config['merge_time_size'], config['ts_size'])

        train_x = np.stack(clean_data, axis=0)
        train_y = np.expand_dims(np.stack((matching_ys), axis=0), axis=1)
        if not excludes is None:
            nondrop_col_idxs = [x for x in range(0, len(train_data.columns)) if train_data.columns[x] not in excludes]
            train_x = train_x[:,:,nondrop_col_idxs]

        categorical_idx = [i for i in range(len(train_data.columns)) if train_data.columns[i] in categorical_feats]
        train_x = data_postproc(train_x, categorical_idx, list(train_data.columns), config)



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

        train_x, train_y = load_mimic_binary_classification(config, path, data_type+"_listfile.csv", data_type, cutoff_seq_len=max_seq_len,
                                                            num_features=num_features, categorical_feats=categoricals, excludes=exclude_list)
        save_data(train_x, x_path)
        save_data(train_y, y_path)

    return train_x, train_y