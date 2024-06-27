import os
from pathlib import Path

import math
import pickle

import pandas as pd
import torch
import numpy as np
import re

HARDCODED_MIMICIII_INITIAL_FEATURES = {
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

GLASGOW_COMASCALE_MAPPING = {
    'Glascow coma scale eye opening': {'Spontaneously': 4, 'To Pressure': 2, 'To Sound': 3, 'None': 1},
    'Glascow coma scale motor response': {'Obeys Commands': 6, 'Localizing': 5, 'Normal Flexion': 4, 'Abnormal Flexion': 3, 'Extension': 2, 'None': 1},
    'Glascow coma scale verbal response': {'Oriented': 5, 'Confused': 4, 'Words': 3, 'Sounds': 2, 'None': 1},
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


def filter_data_check(train_data, cutoff_seq_len):
    passes = True
    # if train_data.shape[0] < cutoff_seq_len:
    #     passes = False
    #
    # if train_data['Hours'].iloc[-1] < 47.0:
    #     passes = False
    return passes


def convert_glascow(row_item):
    try:
        return float(re.sub("[^0-9]", "", str(row_item)))
    except:
        #print("Malformed Categorical Row")
        return 0.0


def data_preproc(train_data, categorical_feats, cutoff_seq_len):
    for cat in categorical_feats:
        cat_col = train_data[cat]
        cat_col = cat_col.apply(convert_glascow)
        train_data[cat] = cat_col

    clean_row = np.expand_dims(train_data.to_numpy(), axis=0)[:, :cutoff_seq_len, :]
    return clean_row



def data_postproc(train_x, categorical_idx, inorder_col_list, config):
    print("STATUS - Starting Data PostProcessing")
    #TODO: confirm cateogrical is exlcude from eman std update
    #TODO: concat with mask
    orig_or_imputed_mask = (train_x != train_x).astype(float)
    for r, row in enumerate(train_x):
        last_feature_vect = [HARDCODED_MIMICIII_INITIAL_FEATURES[x] for x in inorder_col_list if x in HARDCODED_MIMICIII_INITIAL_FEATURES.keys()]
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

    if config['training']['data']['input_concat_w_mask']:
        return np.concatenate((train_x, orig_or_imputed_mask), axis=-1)
    else:
        return train_x


def merge_time_into_windows(train_x, window_size, ts_size, time_index):
    final_train = []
    for icu_stay in train_x:
        merged_stay = []
        icu_stay = icu_stay.to_numpy()

        windows = [(x*window_size,(x+1)*window_size) for x in range(0,int(ts_size))]
        for window in windows:
            matching_timepoint_idxs = np.argwhere(np.logical_and(icu_stay[:,time_index] >= window[0], icu_stay[:,time_index] < window[1]))
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
    print("STATUS - Creating MIMIC-III data")

    preproc_method = config['training']['data']['data_preproc']
    n_hour_per_merge_timepoint = config['training']['data']['merge_time_size']
    n_hours_to_use = config['training']['data']['n_hours_to_use']

    train_file = pd.read_csv(base_path / filename)
    train_stay_ref = train_file["stay"]
    time_index = 0

    notused_seq_lens = []
    used_seq_lens = []

    read_folder = datatype
    if datatype == "val":
        read_folder = "train"

    clean_data = []
    matching_ys = []
    print(f"Using '{preproc_method}' as preprocessing method.")

    for i, stay_ref in enumerate(train_stay_ref):
        train_data = pd.read_csv(base_path / (read_folder+"/"+stay_ref))

        matching_ys.append(train_file["y_true"].iloc[i])

        if preproc_method == 'PaperDescription':
            passes_filter = filter_data_check(train_data, cutoff_seq_len)
            if passes_filter:
                clean_row = data_preproc(train_data, categorical_feats, cutoff_seq_len)
                clean_data.append(clean_row)
                used_seq_lens.append(clean_row.shape[1])
            else:
                notused_seq_lens.append(train_data.shape[0])
        elif preproc_method == 'mimic3benchmark':
            temp = train_data.fillna('')
            temp['Glascow coma scale total'] = temp['Glascow coma scale total'].astype(str).apply(
                lambda x: x.split('.')[0])
            temp = temp.to_numpy()
            clean_data.append(temp)
            used_seq_lens.append(temp.shape[1])
        else:
            for cat in categorical_feats:
                if cat in GLASGOW_COMASCALE_MAPPING.keys():
                    train_data[cat] = train_data[cat].apply(lambda x: GLASGOW_COMASCALE_MAPPING[cat][x] if x in GLASGOW_COMASCALE_MAPPING[cat].keys()  else x).apply(lambda y: re.sub("\D", "", y) if type(y) == str else y)
            train_data = train_data.replace("", np.nan)
            clean_data.append(train_data)
            used_seq_lens.append(train_data.shape[1])

    if preproc_method == 'mimic3benchmark':
        from mimic3models.preprocessing import Normalizer, Discretizer

        discretizer = Discretizer(timestep=float(),
                                  store_masks=True,
                                  impute_strategy='previous',
                                  start_time='zero')

        discretizer_header = discretizer.transform(temp)[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
        normalizer = Normalizer(fields=cont_channels)
        normalizer_state = 'ihm_ts1.0.input_str_previous.start_time_zero.normalizer'
        normalizer.load_params(normalizer_state)

        ts = [n_hours_to_use for i in range(len(clean_data))]
        clean_data = [discretizer.transform(X.astype(str), end=t)[0] for (X, t) in zip(clean_data, ts)]
        if normalizer is not None:
            clean_data = [normalizer.transform(X) for X in clean_data]
        train_x = np.stack(clean_data, axis=0)
        train_y = np.expand_dims(np.stack((matching_ys), axis=0), axis=1)
    else:
        config['ts_size'] = n_hours_to_use // n_hour_per_merge_timepoint
        print(f"STATUS - Data Processing - Merging time into windows")
        clean_data = merge_time_into_windows(clean_data, n_hour_per_merge_timepoint, config['ts_size'], time_index)

        train_x = np.stack(clean_data, axis=0)
        train_y = np.expand_dims(np.stack((matching_ys), axis=0), axis=1)
        if not excludes is None:
            nondrop_col_idxs = [x for x in range(0, len(train_data.columns)) if train_data.columns[x] not in excludes]
            train_x = train_x[:,:,nondrop_col_idxs]
            remaining_col_names = train_data.columns[nondrop_col_idxs]

        categorical_idx = [i for i in range(len(remaining_col_names)) if remaining_col_names[i] in categorical_feats]
        train_x = data_postproc(train_x, categorical_idx, list(remaining_col_names), config)


    remaining_column_names = list(remaining_col_names)
    #remaining_column_names.pop(time_index)

    return train_x, train_y, remaining_column_names


def load_file_data(config, data_type="train"):
    datadir = config['datadir']
    mimic_data_path = config['mimic_data_path']
    categoricals = config['categorical_features']
    exclude_list = config['excludes']
    max_seq_len = config['training']['data']['max_seq_len']
    num_features = config['num_features']
    data_filename = config['data_suffix']

    data_package_path = datadir + data_type+data_filename
    if config['experiment']['load_data']:
        if os.path.isfile(data_package_path):
            print("STATUS - Loading MIMIC-III data from pre-built files")
            data_dict = load_data(data_package_path)
            data_x = data_dict['data_x']
            data_y = data_dict['data_y']
            feature_names = data_dict['feature_names']
        else:
            raise FileNotFoundError(f"Config specifies data should be loaded, but file {data_package_path} not found")
    else:
        mimic_data_path = Path(mimic_data_path)
        data_x, data_y, feature_names = load_mimic_binary_classification(config,
                                                                         mimic_data_path,
                                                                         data_type+"_listfile.csv",
                                                                         data_type,
                                                                         cutoff_seq_len=max_seq_len,
                                                                         num_features=num_features,
                                                                         categorical_feats=categoricals,
                                                                         excludes=exclude_list)

        data_list = {'data_x':data_x,
                     'data_y':data_y,
                     'feature_names':feature_names}
        save_data(data_list, filepath=data_package_path)

    return data_x, data_y, feature_names