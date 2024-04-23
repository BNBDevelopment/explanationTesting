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

def init_autoencoder(configuration, data):
    if configuration['load_autoencoder']:
        model = torch.load(configuration['autoencoder_path'])
    else:
        model = AutoEncoder()
        configuration['optimizer'] = torch.optim.Adam(model.parameters(), lr=configuration['lr'])
        model = train(model, configuration, data['train_x'], data['train_x'], data['val_x'], data['val_x'], manual_loss_fn=torch.nn.MSELoss(), is_autoencoder=True)
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

    #autoencoder = init_autoencoder(configuration, data)

    test_subset_size = configuration['explanation_methods']['n_background_data']
    if test_subset_size > 0 and test_subset_size < 1.0:
        test_subset_size = test_subset_size * test_x.shape[0]
    for_background_idxs = random.sample(range(0, test_x.shape[0]), test_subset_size)
    to_explain_idxs = list(set(range(0, test_x.shape[0])).difference(set(for_background_idxs)))

    x_background = test_x[for_background_idxs]
    y_background = test_y[for_background_idxs]

    x_toExplain = test_x[to_explain_idxs]
    y_toExplain = test_y[to_explain_idxs]

    methods_enabled = {
        "WindowSHAP": {"function": do_WindowSHAP, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
        #"GradCAM": {"function": do_GradCAM, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},

        "CoMTE": {"function": do_COMTE, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
        "NUNCF": {"function": do_NUNCF, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
        "Anchors": {"function": do_Anchors, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},

        "Dynamask": {"function": do_Dynamask, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
        "LORE": {"function": do_LORE, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
    }

    n_explns_per_method = 4
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

    if configuration['gen_presentation_data']:
        toexplain_predictions = ModelWrapper(model).predict(x_toExplain)
        idx_pred1_true1 = np.argwhere(np.logical_and(toexplain_predictions == 1, y_toExplain.flatten() == 1))
        idx_pred1_true0 = np.argwhere(np.logical_and(toexplain_predictions == 1, y_toExplain.flatten() == 0))
        idx_pred0_true1 = np.argwhere(np.logical_and(toexplain_predictions == 0, y_toExplain.flatten() == 1))
        idx_pred0_true0 = np.argwhere(np.logical_and(toexplain_predictions == 0, y_toExplain.flatten() == 0))

        matching_data_subset_x = []
        matching_data_subset_y = []
        for i in range(configuration['presentation_examples_per_category']):
            matching_data_subset_x.append(x_toExplain[idx_pred1_true1[i]])
            matching_data_subset_y.append(y_toExplain[idx_pred1_true1[i]])

            matching_data_subset_x.append(x_toExplain[idx_pred1_true0[i]])
            matching_data_subset_y.append(y_toExplain[idx_pred1_true0[i]])

            matching_data_subset_x.append(x_toExplain[idx_pred0_true1[i]])
            matching_data_subset_y.append(y_toExplain[idx_pred0_true1[i]])

            matching_data_subset_x.append(x_toExplain[idx_pred0_true0[i]])
            matching_data_subset_y.append(y_toExplain[idx_pred0_true0[i]])
        x_toExplain = np.stack(matching_data_subset_x).squeeze()
        y_toExplain = np.reshape(np.array(matching_data_subset_y), (-1,1))

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
                    print(f"Time Taken for {expl_method_name} is {time_seconds_taken}")

                    pickel_results(generated_explanation, f"{explanation_output_folder}explanation.pkl")
                    expl_information["result_store"]["explanations"].append(generated_explanation)
                    expl_information["result_store"]["time_taken"].append(time_seconds_taken)
                    expl_information["result_store"]["random_seed"].append(j)
                    expl_information["result_store"]["samples_explained"].append(sample_to_explain)
                    expl_information["result_store"]["item_index"].append(to_explain_idxs[i_expl])
                    pickel_results(methods_enabled, f"{explanation_output_folder}../../all_explanations.pkl")

                    plt.close('all')
                except Exception as e:
                    if configuration['throw_errors']:
                        raise e
                    else:
                        print(f"EXCEPTION - Exception in {expl_method_name}\n\n")
                        print(e)
                        print(traceback.format_exc())
                        print(f"\n\n")

    methods_enabled["configuration_with_data"] = explanation_config
    pickel_results(methods_enabled, f"{explanation_output_folder}../../all_explanations_data.pkl")

    #run_analysis(model, test_x)

if __name__ == "__main__":
    main()