import random

import numpy as np

from shared.shared_utils import pickel_results
from train.train_utils import ModelWrapper


def update_explanation_outputs(config, expl_information, generated_explanation, explanation_output_folder, time_seconds_taken,
                               rand_seed, sample_to_explain, ordered_indexes, index_to_explain, methods_implemented, experiment_out_path):
    pickel_results(generated_explanation, f"{explanation_output_folder}explanation.pkl")
    expl_information["result_store"]["explanations"].append(generated_explanation)
    expl_information["result_store"]["time_taken"].append(time_seconds_taken)
    expl_information["result_store"]["random_seed"].append(rand_seed)
    expl_information["result_store"]["samples_explained"].append(sample_to_explain)
    if config['class_balance_explanations']:
        expl_information["result_store"]["item_index"].append(ordered_indexes[index_to_explain])
    else:
        expl_information["result_store"]["item_index"].append(index_to_explain)
    pickel_results(methods_implemented, f"{experiment_out_path}/all_explanations.pkl")


def order_data_by_correct_incorrect_and_prediction(config, model, x_toExplain, y_true):
    y_predicted = ModelWrapper(model, skip_autobatch=config['skip_autobatch']).predict_label(x_toExplain)
    y_true = y_true.flatten()

    n_instances_to_explain = config['n_instances_to_explain']
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
    except Exception as ex:
        raise ("FATAL - Not enough samples to keep data balanced!")
    ordered_x_to_explain = np.stack(matching_data_subset_x).squeeze()
    ordered_y_true = np.reshape(np.array(matching_data_subset_y), (-1, 1))

    order_format = ["(ModelPredicts=1 TrueLabel=1)", "(ModelPredicts=1 TrueLabel=0)", "(ModelPredicts=0 TrueLabel=1)", "(ModelPredicts=0 TrueLabel=0)"]
    print("Reordered Data is formatted: \n"
          f"\t{', '.join(order_format)}")
    return ordered_x_to_explain, ordered_y_true, order_format, ordered_indexes


def split_data_for_explanation(configuration, test_x, test_y):
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

    return x_background, y_background, x_to_explain, y_true_to_explain


def build_explanation_config(config, model, feature_names, x_toExplain, x_background, y_background):
    explanation_config = {}
    explanation_config["experiment_config"] = config
    explanation_config["background_data"] = {"x": x_background, "y": y_background}
    explanation_config["model"] = ModelWrapper(model, skip_autobatch=config['skip_autobatch'])
    explanation_config["model_type"] = "lstm"
    explanation_config["feature_names"] = feature_names
    explanation_config["window_length"] = 1
    explanation_config["eval_approach"] = "GRAD"
    explanation_config["what_is_second_dim"] = 'time'
    explanation_config["n_timesteps"] = x_toExplain.shape[1]
    explanation_config["n_features"] = x_toExplain.shape[2]

    for method_dict, m_vals in config['explanation_methods']['methods'].items():
        explanation_config[method_dict] = m_vals
    return explanation_config