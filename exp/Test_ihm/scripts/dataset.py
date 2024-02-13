import math
import pickle

import pandas as pd
import torch
import numpy as np

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