import pickle
import time

import torch
import torch.nn as nn

import ex_dataset
from analysis import run_analysis
from ex_explanation_methods import do_WindowSHAP, do_comte, do_GradCAM, do_COMTE, do_NUNCF
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
        'load_model': False,
        'load_model_path': "_saved_models/TestModel_B_128_3_epoch62_aucroc0.63554.pt"
    }

    train_x, train_y = ex_dataset.load_file_data(configuration, data_type="train")
    val_x, val_y = ex_dataset.load_file_data(configuration, data_type="val")
    do_wrapping_of_model = True
    type_of_wrapped_submodel = 'lstm'

    if configuration['load_model']:
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
    windowshap_res = []
    gradcam_res = []
    comte_res = []
    nuncaf_res = []

    methods_enabled = [
        #"WS",
        #"GCAM",
        "COMTE",
        #"NUNCF"
    ]

    for i in range(11, n_explanation_test):
        to_test_idx = i//10
        set_random_seed(i)

        # WindowSHAP
        if "WS" in methods_enabled:
            try:
                print(f"Running WindowSHAP...")
                windowshap_start = time.time()
                res1 = do_WindowSHAP(model, configuration, train_x, test_x, feature_names=ft, wrap_model=do_wrapping_of_model,
                              model_type=type_of_wrapped_submodel, num_background=50, test_idx=to_test_idx, num_test_samples=1)
                windowshap_end = time.time()
                windowshap_res.append(res1)
                pickel_results(windowshap_res, f"pickel_results/window_shap_results_{i}.pkl")

                print(f"WindowShap runtime: {windowshap_end - windowshap_start} ")
            except Exception as e:
                print(f"\n--FAILED-- in WindowSHAP! {e}")



        # GradCAM
        if "GCAM" in methods_enabled:
            try:
                print(f"Running GCAM...")
                gradcam_start = time.time()
                res2 = do_GradCAM(model, configuration, train_x, test_x, test_y, feature_names=ft, wrap_model=do_wrapping_of_model,
                               model_type=type_of_wrapped_submodel, num_background=50, test_idx=to_test_idx, num_test_samples=1)
                gradcam_end = time.time()
                gradcam_res.append(res2)
                pickel_results(gradcam_res, f"pickel_results/gradcam_results_{i}.pkl")

                print(f"GradCAM runtime: {gradcam_end - gradcam_start} ")
            except Exception as e:
                print(f"\n--FAILED-- in GradCAM! {e}")
                raise ValueError

        #COMTE
        if "COMTE" in methods_enabled:
            try:
                print(f"Running COMTE...")
                comte_start = time.time()
                res3 = do_COMTE(model, configuration, train_x, test_x, test_y, feature_names=ft, wrap_model=do_wrapping_of_model,
                               model_type=type_of_wrapped_submodel, num_background=50, test_idx=to_test_idx, num_test_samples=1)
                comte_end = time.time()
                comte_res.append(res3)
                pickel_results(comte_res, f"pickel_results/comte_results_{i}.pkl")

                print(f"COMTE runtime: {comte_end - comte_start} ")
            except Exception as e:
                print(f"\n--FAILED-- in CoMTE! {e}")

        #NUNCF
        if "NUNCF" in methods_enabled:
            try:
                print(f"Running NUNCF...")
                nuncf_start = time.time()
                res4 = do_NUNCF(model, configuration, train_x, test_x, test_y, feature_names=ft, wrap_model=do_wrapping_of_model,
                               model_type=type_of_wrapped_submodel, num_background=50, test_idx=to_test_idx, num_test_samples=1)
                nuncf_end = time.time()
                nuncaf_res.append(res4)
                pickel_results(nuncaf_res, f"pickel_results/nuncaf_results_{i}.pkl")

                print(f"NUNCF runtime: {nuncf_end - nuncf_start} ")
            except Exception as e:
                print(f"\n--FAILED-- in NUNCF! {e}")

    run_analysis(model, test_x)

if __name__ == "__main__":
    main()