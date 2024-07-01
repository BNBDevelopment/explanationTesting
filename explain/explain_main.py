import pathlib
import time

import numpy
import torch
from matplotlib import pyplot as plt

import traceback

import numpy as np
import tensorflow as tf

from explain.explain_explain import do_WindowSHAP, do_COMTE, do_Anchors
from explain.explain_utils import build_explanation_config, split_data_for_explanation, \
    order_data_by_correct_incorrect_and_prediction, update_explanation_outputs
from shared.shared_utils import set_random_seed, initialize_configuration, pickel_results
from train.train_data import load_file_data


def get_implemented_methods():
    #List of the methods currently implemented for use
    impl_dict = {
        ############################# Attribution Methods
        "WindowSHAP": {"function": do_WindowSHAP, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index": [], "samples_explained": []}},
        #"GradCAM": {"function": do_GradCAM, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
        # "Dynamask": {"function": do_Dynamask, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},

        ############################# Counterfactual Methods
        "CoMTE": {"function": do_COMTE, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index": [], "samples_explained": []}},
        #"NUNCF": {"function": do_NUNCF, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},

        ############################# Rule-based Methods
        "Anchors": {"function": do_Anchors, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index": [], "samples_explained": []}},
        #"AnchorsAlt": {"function": do_AnchorsAlt, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},

        ############################# Natural Language Methods
        #"LORE": {"function": do_LORE, "result_store": {"explanations": [], "time_taken": [], "random_seed": [], "item_index":[], "samples_explained":[]}},
    }
    return impl_dict


class TensorflowWrapper:
    def __init__(self, raw_model, device):
        self.torch_device = device
        self.m = raw_model
        if ':' in device:
            d_type, d_num = device.split(":")
            if d_type.lower() == 'cuda':
                d_type = "/GPU"
            else:
                d_type = "/CPU"
            self.device = f"{d_type}:{d_num}"
        else:
            if 'cuda' in device:
                self.device = '/GPU'
            else:
                self.device = '/CPU'
        self.n_classes = 2

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            with tf.device(self.device):
                x = tf.convert_to_tensor(x)
        elif isinstance(x, numpy.ndarray):
            with tf.device(self.device):
                x = tf.convert_to_tensor(x)
        else:
            raise NotImplementedError

        with tf.device(self.device):
            tf_out = self.m(x)
        outs = tf_out._numpy()
        return torch.from_numpy(outs).to(self.torch_device)

def main():
    print("STATUS - Initializing Configuration")
    config = initialize_configuration()
    set_random_seed(12345)

    print("STATUS - Starting Initial Data Load")
    train_x, train_y, feature_names = load_file_data(config, data_type="train")
    val_x, val_y, _ = load_file_data(config, data_type="val")
    test_x, test_y, _ = load_file_data(config, data_type="test")

    #Load the model
    if config['experiment']['load_model_path'][-3:] == '.h5':
        raw_model = tf.keras.models.load_model(config['experiment']['load_model_path'])
        model = TensorflowWrapper(raw_model, device=config['device'])
    else:
        model = torch.load(config['experiment']['load_model_path'], map_location=torch.device(config['device']))



    #Load the data
    x_background, y_background, x_explained, y_true_explained = split_data_for_explanation(config, test_x, test_y)

    #Builds the explanation config, which is the config common to all of the explanation methods.
    explanation_config = build_explanation_config(config, model, feature_names, x_explained, x_background, y_background)

    #Get the available/implemented expl;anation methods
    available_exp_methods = get_implemented_methods()

    #Select which of the methods to use based on the configuration
    methods_to_use = {m:v for m, v in available_exp_methods.items() if m.lower() in [x.lower() for x in config['explanation_methods']['methods'].keys()]}

    # Reorders the data so that we see explanations for every prediction-label case (i.e. Index 1 = "Model predicts 1 but True Label is 0", Index 2 = "Model predicts 1 but True Label is 1", etc...)
    if config['class_balance_explanations']:
        x_explained, y_true_explained, order_format, ordered_indexes = order_data_by_correct_incorrect_and_prediction(config, model, x_explained, y_true_explained)


    restarting_index = 0
    n_methods_to_use = len(methods_to_use)
    n_instances_to_explain = config['n_instances_to_explain']
    n_rand_seed_to_try = config['n_rand_seed_to_try']
    n_trials_per_rand_seed = config['n_trials_per_rand_seed']
    print(f"Number of Methods: {n_methods_to_use}\n"
          f"Number of instances to explain: {n_instances_to_explain}\n"
          f"Number of Random Seeds to try: {n_rand_seed_to_try}\n"
          f"Number of trials per Random Seed: {n_trials_per_rand_seed}\n"
          f"--------------------------------------\n"
          f"Total Number of Explanations generated: {n_methods_to_use*n_instances_to_explain*n_rand_seed_to_try*n_trials_per_rand_seed}")

    experiment_out_path = pathlib.Path(config['save_explanation_path']) / config['experiment']['name']

    #Loop that iterates over the experiments
    for index_to_explain in range(restarting_index, n_instances_to_explain):
        for rand_seed in range(n_rand_seed_to_try):
            set_random_seed(rand_seed)
            for trial_num in range(n_trials_per_rand_seed):
                sample_to_explain = {"x": np.expand_dims(x_explained[index_to_explain], 0),
                                     "y": y_true_explained[index_to_explain]}

                for expl_method_name, expl_information in methods_to_use.items():
                    print(f"{expl_method_name} Explanation Method Starting for iteration {index_to_explain}")
                    try:
                        #explanation_config["model"].model.to(config["device"])
                        explanation_function = expl_information["function"]

                        #Create the explanation output path
                        explanation_output_folder = experiment_out_path / f"/{expl_method_name}/instance-{index_to_explain}_random_seed-{rand_seed}_trial-{trial_num}/"
                        path = pathlib.Path(explanation_output_folder)
                        path.mkdir(parents=True, exist_ok=True)

                        #Generate the explanation and start the timer
                        start_time = time.time()
                        generated_explanation = explanation_function(explanation_config,
                                                                     sample_to_explain,
                                                                     explanation_output_folder)
                        end_time = time.time()

                        time_seconds_taken = end_time-start_time
                        print(f"Time Taken for {expl_method_name} is {time_seconds_taken}")

                        update_explanation_outputs(config,
                                                   expl_information,
                                                   generated_explanation,
                                                   explanation_output_folder,
                                                   time_seconds_taken,
                                                   rand_seed,
                                                   sample_to_explain,
                                                   ordered_indexes,
                                                   index_to_explain,
                                                   available_exp_methods,
                                                   experiment_out_path)

                        plt.close('all')
                    except Exception as e:
                        if config['reporting']['throw_errors']:
                            raise e
                        else:
                            print(f"EXCEPTION - Exception in {expl_method_name}\n")
                            print(e)
                            print(traceback.format_exc())
                            print(f"\n")

    available_exp_methods["configuration_with_data"] = explanation_config
    pickel_results(available_exp_methods, f"{explanation_output_folder}../../all_explanations_data.pkl")

    #run_analysis(model, test_x)

if __name__ == "__main__":
    main()
