import pickle
import time

import torch
import torch.nn as nn

import ex_dataset
from ex_explanation_methods import do_WindowSHAP, do_comte, do_GradCAM, do_COMTE, do_NUNCF, do_TimeSHAP
from ex_models import V1Classifier, BasicLSTM, select_model
from ex_train import train

import random
import numpy as np

def set_random_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
set_random_seed(12345)


def pickel_results(obj, fname):
    pfile = open(fname, "wb")
    pickle.dump(obj, pfile)
    pfile.close()

def main():
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.0001
    cutoff_seq_len = 70
    excludes = ['Glascow coma scale eye opening', 'Glascow coma scale motor response',
                'Glascow coma scale verbal response']
    num_features = 18 - len(excludes)
    n_classes = 2

    loss_weights = torch.ones((n_classes))
    loss_weights[1] = 1.5

    configuration = {
        'n_epochs': 100,  # default from their code
        'lr': lr,
        'batch_size': 32,  # their code says is best
        'device': current_device,
        'loss_fn': nn.BCELoss(loss_weights).to(current_device),  # default from their code
        #'loss_fn': nn.NLLLoss(loss_weights),
        'save_model_path': "_saved_models/",
        'model_name': "TestModel_B_128_3",
        'excluded_features_list': excludes,
        'mimic_data_path': "./data/in-hospital-mortality/",
        'datadir': "./data/my_mimic/",
        'max_seq_len': cutoff_seq_len,
        'data_filename': "_ihm_30",
        'num_classes': n_classes,
        'num_features': num_features,
        'model_n_layers': 5,
        'model_dropout': 0.1,
        'model_bias': True,
        'model_bidirectional': True,
        'model_hdim': 256,
        'load_model_path': "_saved_models/TestModel_B_128_3_epoch62_aucroc0.63554.pt"
    }

    train_x, train_y = ex_dataset.load_file_data(configuration, data_type="train")
    val_x, val_y = ex_dataset.load_file_data(configuration, data_type="val")
    do_wrapping_of_model = True
    type_of_wrapped_submodel = 'lstm'

    if 'load_model_path' in configuration.keys():
        model = torch.load(configuration['load_model_path'])
    else:
        model_type = "benchmark_lstm"
        #model_type = "channelwise_lstm"

        model = select_model(model_type, configuration)
        configuration['optimizer'] = torch.optim.Adam(model.parameters(), lr=lr)  # from paper

        model = train(model, configuration, train_x, train_y, val_x, val_y)


    #test_x, test_y = load_mimic_binary_classification(path, "test_listfile.csv", "test", cutoff_seq_len=cutoff_seq_len, num_features=num_features, categorical_feats=excludes)
    test_x, test_y = ex_dataset.load_file_data(configuration, data_type="test")

    test_subset_size = 300
    test_x = test_x[:test_subset_size, :, :]
    test_y = test_y[:test_subset_size]

    ft = ["Hours","Capillary refill rate","Diastolic blood pressure","Fraction inspired oxygen",
          "Glascow coma scale total","Glucose","Heart Rate","Height","Mean blood pressure","Oxygen saturation",
          "Respiratory rate","Systolic blood pressure","Temperature","Weight","pH"]


    #COMTE code only works for univariate?
    #do_comte(model, train_x, train_y, test_x, test_y, test_id_to_explain=0)


    n_explanation_test = 100
    timeshap_res = []
    gradcam_res = []
    comte_res = []
    nuncaf_res = []

    for i in range(1, n_explanation_test):
        to_test_idx = i//10
        set_random_seed(i)

        timeshap_start = time.time()
        res1 = do_TimeSHAP(model, configuration, train_x, test_x, feature_names=ft, wrap_model=do_wrapping_of_model,
                      model_type=type_of_wrapped_submodel, num_background=50, test_idx=to_test_idx, num_test_samples=1)
        timeshap_end = time.time()
        timeshap_res.append(res1)
        pickel_results(timeshap_res, f"pickel_results/time_shap_results_{i}.pkl")

        print(f"TimeShap runtime: {timeshap_end - timeshap_start} ")


if __name__ == "__main__":
    main()