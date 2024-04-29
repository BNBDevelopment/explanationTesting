import argparse
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
from ex_explanation_methods import do_WindowSHAP, do_GradCAM, do_COMTE, do_NUNCF, do_Anchors, do_Dynamask, do_LORE
from ex_models import V1Classifier, BasicLSTM, select_model, AutoEncoder
from ex_train import train
import traceback
import matplotlib

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
    parser = argparse.ArgumentParser(
        prog='Explanation Testing Framework',
        description='Run a selection of explanation methods')
    parser.add_argument('configuration')
    args = parser.parse_args()
    config_path = args.configuration

    stream = open(config_path, "r")
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
        model = torch.load(configuration['load_model_path'], map_location=torch.device('cpu'))
    else:
        model = select_model(configuration)
        configuration['optimizer'] = torch.optim.Adam(model.parameters(), lr=configuration['lr'])
        model = train(model, configuration, data['train_x'], data['train_y'], data['val_x'], data['val_y'])
    return model

def init_autoencoder(configuration, data):
    if configuration['load_autoencoder']:
        model = torch.load(configuration['autoencoder_path'])
    else:
        model = AutoEncoder()
        configuration['optimizer'] = torch.optim.Adam(model.parameters(), lr=configuration['lr'])
        model = train(model, configuration, data['train_x'], data['train_x'], data['val_x'], data['val_x'], manual_loss_fn=torch.nn.MSELoss(), is_autoencoder=True)
    return model


def order_data_by_correct_incorrect_and_prediction(configuration, model, x_toExplain, y_true):
    y_predicted = ModelWrapper(model).predict(x_toExplain)
    y_true = y_true.flatten()

    n_instances_to_explain = configuration['n_instances_to_explain']
    num_cases = 4
    print(f"Number of prediction-label combination cases is {num_cases}")

    idx_pred1_true1 = np.argwhere(np.logical_and(y_predicted == 1, y_true == 1))
    idx_pred1_true0 = np.argwhere(np.logical_and(y_predicted == 1, y_true == 0))
    idx_pred0_true1 = np.argwhere(np.logical_and(y_predicted == 0, y_true == 1))
    idx_pred0_true0 = np.argwhere(np.logical_and(y_predicted == 0, y_true == 0))
    cases_list = [idx_pred1_true1, idx_pred1_true0, idx_pred0_true1, idx_pred0_true0]

    matching_data_subset_x = []
    matching_data_subset_y = []

    assert n_instances_to_explain % num_cases == 0, f"FATAL - n_instances_to_explain {n_instances_to_explain} is not evenly divisible by num_cases {num_cases}"
    num_cases_per_pl_combo = n_instances_to_explain // num_cases

    ordered_indexes = []

    try:
        for idx_combo in range(num_cases_per_pl_combo):
            for list_case_idxs in cases_list:
                matching_data_subset_x.append(x_toExplain[list_case_idxs[idx_combo]])
                matching_data_subset_y.append(y_true[list_case_idxs[idx_combo]])
                ordered_indexes.append(list_case_idxs[idx_combo].item())
    except:
        raise ("FATAL - Not enough samples to keep data balanced!")
    ordered_x_to_explain = np.stack(matching_data_subset_x).squeeze()
    ordered_y_true = np.reshape(np.array(matching_data_subset_y), (-1, 1))

    order_format = ["(ModelPredicts=1 TrueLabel=1)", "(ModelPredicts=1 TrueLabel=0)", "(ModelPredicts=0 TrueLabel=1)", "(ModelPredicts=0 TrueLabel=0)"]
    print("Reordered Data is formatted: \n"
          f"\t{', '.join(order_format)}")
    return ordered_x_to_explain, ordered_y_true, order_format, ordered_indexes


def build_explanation_config(configuration, model, feature_names, x_toExplain, x_background, y_background):
    explanation_config = {}
    explanation_config["experiment_config"] = configuration
    explanation_config["background_data"] = {"x": x_background, "y": y_background}
    explanation_config["model"] = ModelWrapper(model)
    explanation_config["model_type"] = "lstm"
    explanation_config["feature_names"] = feature_names
    explanation_config["window_length"] = 1
    explanation_config["eval_approach"] = "GRAD"
    explanation_config["what_is_second_dim"] = 'time'
    explanation_config["n_timesteps"] = x_toExplain.shape[1]
    explanation_config["n_features"] = x_toExplain.shape[2]

    for method_dict, m_vals in configuration['explanation_methods']['methods'].items():
        explanation_config[method_dict] = m_vals
    return explanation_config

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
    #autoencoder = init_autoencoder(configuration, data)


    num_background_samples = configuration['explanation_methods']['num_background_samples']
    if num_background_samples > 0 and num_background_samples < 1.0:
        num_background_samples = int(num_background_samples * test_x.shape[0] // 1)
    indexes_for_background = random.sample(range(0, test_x.shape[0]), num_background_samples)
    indexes_to_explain = list(set(range(0, test_x.shape[0])).difference(set(indexes_for_background)))

    # Splits the data into data (instances) to explain, and data that can be used as a background. Some methods require background data against which to compare the instances that are being explained.
    x_background = test_x[indexes_for_background]
    y_background = test_y[indexes_for_background]
    x_to_explain = test_x[indexes_to_explain]
    y_true_to_explain = test_y[indexes_to_explain]

    #List of the methods currently implemented for use
    methods_implemented = {
        # Attribution Methods
        "WindowSHAP": {"function": do_WindowSHAP, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
        "GradCAM": {"function": do_GradCAM, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
        # Counterfactual Methods
        "CoMTE": {"function": do_COMTE, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
        "NUNCF": {"function": do_NUNCF, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
        "Anchors": {"function": do_Anchors, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
        # Rule-based Methods
        "Dynamask": {"function": do_Dynamask, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
        # Natural Language Methods
        "LORE": {"function": do_LORE, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
    }

    #Builds the explanation configuration, which is the config common to all of the explanation methods
    explanation_config = build_explanation_config(configuration, model, feature_names, x_to_explain, x_background, y_background)

    # Reorders the data so that we see explanations for every prediction-label case (i.e. Index 1 = "Model predicts 1 but True Label is 0", Index 2 = "Model predicts 1 but True Label is 1", etc...)
    x_to_explain, y_true_to_explain, order_format, ordered_indexes = order_data_by_correct_incorrect_and_prediction(configuration, model, x_to_explain, y_true_to_explain)

    #Decides if we are going to show the matplotlib visualization in a new window (the default one that matplotlib/pyplot spawns)
    if not configuration['halt_and_show_matplotlib']:
        plt.ioff()
        matplotlib.use('Agg')

    #Select which of the XAI methods we are going to use based on their presence in the configuration file
    methods_to_use = {}
    for m, v in methods_implemented.items():
        if m.lower() in [x.lower() for x in configuration['explanation_methods']['methods'].keys()]:
            methods_to_use[m] = v


    restarting_index = 0
    n_methods_to_use = len(methods_to_use)
    n_instances_to_explain = configuration['n_instances_to_explain']
    n_rand_seed_to_try = configuration['n_rand_seed_to_try']
    n_trials_per_rand_seed = configuration['n_trials_per_rand_seed']
    print(f"Number of Methods: {n_methods_to_use}\n"
          f"Number of instances to explain: {n_instances_to_explain}\n"
          f"Number of Random Seeds to try: {n_rand_seed_to_try}\n"
          f"Number of trials per Random Seed: {n_trials_per_rand_seed}\n"
          f"--------------------------------------\n"
          f"Total Number of Explanations generated: {n_methods_to_use*n_instances_to_explain*n_rand_seed_to_try*n_trials_per_rand_seed}")

    experiment_out_path = configuration['save_model_path'] + configuration['model_name']

    for index_to_explain in range(restarting_index, n_instances_to_explain):
        for rand_seed in range(n_rand_seed_to_try):
            set_random_seed(rand_seed)
            for trial_num in range(n_trials_per_rand_seed):
                sample_to_explain = {"x": np.expand_dims(x_to_explain[index_to_explain], 0), "y": y_true_to_explain[index_to_explain]}

                for expl_method_name, expl_information in methods_to_use.items():
                    print(f"{expl_method_name} Explanation Method Starting for iteration {index_to_explain}")
                    try:
                        explanation_config["model"].model.to(configuration["device"])
                        explanation_function = expl_information["function"]

                        explanation_output_folder = experiment_out_path + f"/{expl_method_name}/instance-{index_to_explain}_random_seed-{rand_seed}_trial-{trial_num}/"
                        path = pathlib.Path(explanation_output_folder)
                        path.mkdir(parents=True, exist_ok=True)

                        start_time = time.time()
                        generated_explanation = explanation_function(explanation_config, sample_to_explain, explanation_output_folder)
                        end_time = time.time()
                        time_seconds_taken = end_time-start_time
                        print(f"Time Taken for {expl_method_name} is {time_seconds_taken}")

                        pickel_results(generated_explanation, f"{explanation_output_folder}explanation.pkl")
                        expl_information["result_store"]["explanations"].append(generated_explanation)
                        expl_information["result_store"]["time_taken"].append(time_seconds_taken)
                        expl_information["result_store"]["random_seed"].append(rand_seed)
                        expl_information["result_store"]["samples_explained"].append(sample_to_explain)
                        expl_information["result_store"]["item_index"].append(ordered_indexes[index_to_explain])
                        pickel_results(methods_implemented, f"{experiment_out_path}/all_explanations.pkl")

                        plt.close('all')
                    except Exception as e:
                        if configuration['throw_errors']:
                            raise e
                        else:
                            print(f"EXCEPTION - Exception in {expl_method_name}\n")
                            print(e)
                            print(traceback.format_exc())
                            print(f"\n")

    methods_implemented["configuration_with_data"] = explanation_config
    pickel_results(methods_implemented, f"{explanation_output_folder}../../all_explanations_data.pkl")

    #run_analysis(model, test_x)

if __name__ == "__main__":
    main()