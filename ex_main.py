import pathlib
import pickle
import time

import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from yaml import CLoader

import ex_dataset
from analysis import run_analysis
from ex_explanation_methods import do_WindowSHAP, do_comte, do_GradCAM, do_COMTE, do_NUNCF, do_Anchors
from ex_models import V1Classifier, BasicLSTM, select_model
from ex_train import train
import traceback

import random
import numpy as np

from ex_utils import ModelWrapper


def set_random_seed(seed_val):
    print(f"Init - Setting random seed to {seed_val}")
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)


def pickel_results(obj, fname):
    pfile = open(fname, "wb")
    pickle.dump(obj, pfile)
    pfile.close()


def initialize_configuration():
    stream = open("config.yaml", "r")
    try:
        config = yaml.load(stream,  Loader=CLoader)
    finally:
        stream.close()

    config['max_seq_len'] = config['cutoff_seq_len']
    config['num_classes'] = 2
    if not config['excludes'] is None:
        config['num_features'] = 18 - len(config['excludes'])
    else:
        config['num_features'] = 18
    config['n_classes'] = 2

    if config['loss_type'] == "NLL":
        config['loss_fn'] = nn.NLLLoss()
    elif config['loss_type'] == "BCE":
        config['loss_fn'] = nn.BCELoss()
    else:
        raise NotImplementedError(f"Loss type {config['loss_type']} is not implemented!")

    return config


def initialize_model(configuration, data):
    if configuration['load_model']:
        model = torch.load(configuration['load_model_path'])
    else:
        model = select_model(configuration)
        configuration['optimizer'] = torch.optim.Adam(model.parameters(), lr=configuration['lr'])
        model = train(model, configuration, data['train_x'], data['train_y'], data['val_x'], data['val_y'])
    return model


def main():
    set_random_seed(12345)
    print("STATUS - Initializing Configuration")
    configuration = initialize_configuration()

    print("STATUS - Starting Initial Data Load")
    train_x, train_y, feature_names = ex_dataset.load_file_data(configuration, data_type="train")
    val_x, val_y, _ = ex_dataset.load_file_data(configuration, data_type="val")
    test_x, test_y, _ = ex_dataset.load_file_data(configuration, data_type="test")
    data = {
        "train_x":train_x,
        "train_y":train_y,
        "val_x":val_x,
        "val_y":val_y,
        "test_x":test_x,
        "test_y":test_y,
    }

    model = initialize_model(configuration, data)

    test_subset_size = configuration['explanation_methods']['n_background_data']
    if test_subset_size > 0 and test_subset_size < 1.0:
        test_subset_size = test_subset_size * test_x.shape[0]
    for_background_idxs = random.sample(range(0, test_x.shape[0]), test_subset_size)
    to_explain_idxs = list(set(range(0, test_x.shape[0])).difference(set(for_background_idxs)))

    x_background = test_x[for_background_idxs]
    y_background = test_y[for_background_idxs]

    x_toExplain = test_x[to_explain_idxs]
    y_toExplain = test_y[to_explain_idxs]

    #n_explanation_test = 100
    # windowshap_res = []
    # gradcam_res = []
    # comte_res = []
    # nuncaf_res = []

    methods_enabled = {
        "WindowSHAP": {"function": do_WindowSHAP, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
        #"GradCAM": {"function": do_GradCAM, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
        #"CoMTE": {"function": do_COMTE, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
        #"NUNCF": {"function": do_NUNCF, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
        #"Anchors": {"function": do_Anchors, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
    }


    n_explns_per_method = 100
    n_explns_per_random_seed = 1

    explanation_config = {}
    explanation_config["experiment_config"] = configuration
    explanation_config["background_data"] = {"x":x_background, "y":y_background}
    explanation_config["model"] = ModelWrapper(model)
    explanation_config["model_type"] = "lstm"
    explanation_config["feature_names"] = feature_names
    explanation_config["window_length"] = 1
    explanation_config["eval_approach"] = "GRAD"
    explanation_config["what_is_second_dim"] = 'time'
    explanation_config["n_timesteps"] = x_toExplain.shape[1]
    explanation_config["n_features"] = x_toExplain.shape[2]

    plt.ioff()
    import matplotlib
    matplotlib.use('Agg')

    starting_index = 0
    for j in range(n_explns_per_random_seed):
        set_random_seed(j)
        for i_expl in range(starting_index, n_explns_per_method):
            sample_to_explain = {"x": np.expand_dims(x_toExplain[i_expl], 0), "y": y_toExplain[i_expl]}

            for expl_method_name, expl_information in methods_enabled.items():
                print(f"{expl_method_name} Explanation Method Starting for iteration {i_expl}")
                try:
                    explanation_config["model"].model.to(configuration["device"])
                    explanation_function = expl_information["function"]
                    #explanation_results = expl_information["result_store"]

                    explanation_output_folder = configuration['save_model_path'] + configuration['model_name'] + f"/{expl_method_name}/RS{j}_Item{i_expl}/"
                    path = pathlib.Path(explanation_output_folder)
                    path.mkdir(parents=True, exist_ok=True)

                    start_time = time.time()
                    generated_explanation = explanation_function(explanation_config, sample_to_explain, explanation_output_folder)
                    end_time = time.time()
                    time_seconds_taken = end_time-start_time

                    pickel_results(generated_explanation, f"{explanation_output_folder}explanation.pkl")
                    expl_information["result_store"]["explanations"].append(generated_explanation)
                    expl_information["result_store"]["time_taken"].append(time_seconds_taken)
                    expl_information["result_store"]["random_seed"].append(j)
                    expl_information["result_store"]["samples_explained"].append(sample_to_explain)
                    expl_information["result_store"]["item_index"].append(to_explain_idxs[i_expl])
                    pickel_results(methods_enabled, f"{explanation_output_folder}../../all_explanations.pkl")

                    plt.close('all')
                except Exception as e:
                    print(f"EXCEPTION - Exception in {expl_method_name}\n\n")
                    print(e)
                    print(traceback.format_exc())
                    print(f"\n\n")

    methods_enabled["configuration_with_data"] = explanation_config
    pickel_results(methods_enabled, f"{explanation_output_folder}../../all_explanations_data.pkl")

    # for i in range(99, n_explanation_test):
    #     to_test_idx = i//10
    #     set_random_seed(i)
    #
    #     # # WindowSHAP
    #     # if "WS" in methods_enabled:
    #     #     try:
    #     #         print(f"Running WindowSHAP...")
    #     #         windowshap_start = time.time()
    #     #         res1 = do_WindowSHAP(model, configuration, train_x, test_x, feature_names=feature_names, wrap_model=do_wrapping_of_model,
    #     #                       model_type=type_of_wrapped_submodel, num_background=50, test_idx=to_test_idx, num_test_samples=1)
    #     #         windowshap_end = time.time()
    #     #         windowshap_res.append(res1)
    #     #         pickel_results(windowshap_res, f"pickel_results/window_shap_results_{i}.pkl")
    #     #
    #     #         print(f"WindowShap runtime: {windowshap_end - windowshap_start} ")
    #     #     except Exception as e:
    #     #         print(f"\n--FAILED-- in WindowSHAP! {e}")
    #     #
    #     #
    #     #
    #     # # GradCAM
    #     # if "GCAM" in methods_enabled:
    #     #     try:
    #     #         print(f"Running GCAM...")
    #     #         gradcam_start = time.time()
    #     #         res2 = do_GradCAM(model, configuration, train_x, test_x, test_y, feature_names=feature_names, wrap_model=do_wrapping_of_model,
    #     #                        model_type=type_of_wrapped_submodel, num_background=50, test_idx=to_test_idx, num_test_samples=1)
    #     #         gradcam_end = time.time()
    #     #         gradcam_res.append(res2)
    #     #         pickel_results(gradcam_res, f"pickel_results/gradcam_results_{i}.pkl")
    #     #
    #     #         print(f"GradCAM runtime: {gradcam_end - gradcam_start} ")
    #     #     except Exception as e:
    #     #         print(f"\n--FAILED-- in GradCAM! {e}")
    #     #         raise ValueError
    #
    #     #COMTE
    #     if "COMTE" in methods_enabled:
    #         try:
    #             print(f"Running COMTE...")
    #             comte_start = time.time()
    #             res3 = do_COMTE(model, configuration, train_x, test_x, test_y, feature_names=feature_names, wrap_model=do_wrapping_of_model,
    #                            model_type=type_of_wrapped_submodel, num_background=50, test_idx=to_test_idx, num_test_samples=1)
    #             comte_end = time.time()
    #             comte_res.append(res3)
    #             pickel_results(comte_res, f"pickel_results/comte_results_{i}.pkl")
    #
    #             print(f"COMTE runtime: {comte_end - comte_start} ")
    #         except Exception as e:
    #             raise e
    #             print(f"\n--FAILED-- in CoMTE! {e}")
    #
    #     #NUNCF
    #     if "NUNCF" in methods_enabled:
    #         try:
    #             print(f"Running NUNCF...")
    #             nuncf_start = time.time()
    #             res4 = do_NUNCF(model, configuration, train_x, test_x, test_y, feature_names=ft, wrap_model=do_wrapping_of_model,
    #                            model_type=type_of_wrapped_submodel, num_background=50, test_idx=to_test_idx, num_test_samples=1)
    #             nuncf_end = time.time()
    #             nuncaf_res.append(res4)
    #             pickel_results(nuncaf_res, f"pickel_results/nuncaf_results_{i}.pkl")
    #
    #             print(f"NUNCF runtime: {nuncf_end - nuncf_start} ")
    #         except Exception as e:
    #             raise e
    #             print(f"\n--FAILED-- in NUNCF! {e}")
    #
    #     if "ANCH" in methods_enabled:
    #         # try:
    #         print(f"Running Anchors...")
    #         anchors_start = time.time()
    #         res1 = do_Anchors(model, configuration, train_x, test_x, feature_names=ft, wrap_model=do_wrapping_of_model,
    #                       model_type=type_of_wrapped_submodel, num_background=50, test_idx=to_test_idx, num_test_samples=1)
    #         anchors_end = time.time()
    #         anchors_res.append(res1)
    #         pickel_results(anchors_res, f"pickel_results/anchors_results_{i}.pkl")
    #
    #         print(f"Anchors runtime: {anchors_end - windowshap_start} ")
    #         # except Exception as e:
    #         #     print(f"\n--FAILED-- in Anchors! {e}")

    run_analysis(model, test_x)

if __name__ == "__main__":
    main()