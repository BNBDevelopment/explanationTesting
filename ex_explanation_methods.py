import copy
import threading

import pandas as pd
import torch
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR
from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF import NativeGuideCF
from anchor import anchor_tabular, anchor_image, anchor_text

from windowshap.windowshap import StationaryWindowSHAP

# from _required_Packages.ForkDynamask.attribution.mask import Mask
# from _required_Packages.ForkDynamask.attribution.perturbation import GaussianBlur

from TSInterpret.InterpretabilityModels.counterfactual.COMTECF import COMTECF

from ex_analysis import plot_original_overlap_counterfactual, plot_original_line_with_vals
import numpy as np
import shap

# from explainers.lore_explainer import LoreTabularExplainer, LoreTabularExplanation
# from externals.LOREM.lorem import LOREM
#from rulematrix import Surrogate


#################################################################################

def do_WindowSHAP(explanation_config, sample_to_explain, explanation_output_folder):
    method_name = "windowshap"
    background_data = explanation_config["background_data"]
    background_data['x'] = background_data['x'][:explanation_config[method_name]['num_background_used']]
    background_data['y'] = background_data['y'][:explanation_config[method_name]['num_background_used']]
    model = explanation_config["model"]
    model_type = explanation_config["model_type"]
    feature_names = explanation_config["feature_names"]
    window_length = explanation_config[method_name]["window_len"]

    if sample_to_explain['x'].shape[1] % window_length != 0:
        raise NotImplementedError(f"FATAL - time series length {sample_to_explain['x'].shape[1]} must be divisible by the window size {window_length}")

    gtw = StationaryWindowSHAP(model, window_length, B_ts=background_data['x'], test_ts=sample_to_explain['x'], model_type=model_type)

    gtw.explainer = shap.KernelExplainer(gtw.wraper_predict, gtw.background_data)
    shap_values = gtw.explainer.shap_values(gtw.test_data)
    shap_values = np.array(shap_values)

    sv = np.repeat(shap_values.flatten().reshape((-1, gtw.num_window)), window_length, axis=1)

    plot_original_line_with_vals(sample_to_explain, sv.transpose(), feature_names, explanation_output_folder)
    return sv


def do_GradCAM(explanation_config, sample_to_explain, explanation_output_folder):
    model = explanation_config["model"]
    feature_names = explanation_config["feature_names"]
    cur_device = explanation_config["experiment_config"]["device"]

    eval_approach = explanation_config["eval_approach"]
    what_is_second_dim = explanation_config["what_is_second_dim"]
    n_timesteps = explanation_config["n_timesteps"]
    n_features = explanation_config["n_features"]
    unwrapped_model = model.model
    unwrapped_model.train()
    explainer_method = TSR(unwrapped_model, NumTimeSteps=n_timesteps, NumFeatures=n_features, method=eval_approach, mode=what_is_second_dim, device=cur_device)

    exp = explainer_method.explain(sample_to_explain['x'], labels=sample_to_explain['y'], TSR=True)

    plot_original_line_with_vals(sample_to_explain, exp, feature_names, explanation_output_folder)
    unwrapped_model.eval()
    return exp


def do_COMTE(explanation_config, sample_to_explain, explanation_output_folder):
    method_name = "comte"
    model = explanation_config["model"]
    feature_names = explanation_config["feature_names"]
    what_is_second_dim = explanation_config["what_is_second_dim"]
    background_data = explanation_config["background_data"]
    background_data['x'] = background_data['x'][:explanation_config[method_name]['num_background_used']]
    background_data['y'] = background_data['y'][:explanation_config[method_name]['num_background_used']]
    cur_device = explanation_config["experiment_config"]["device"]
    cf_feats_to_switch = explanation_config[method_name]["max_n_feats"]
    tau_confidence = explanation_config[method_name]['tau_confidence']

    unwrapped_model = model.model
    unwrapped_model.eval()

    comte_formatted_data = tuple([background_data['x'], background_data['y'].squeeze()])
    explainer_method = COMTECF(unwrapped_model.to('cpu'), comte_formatted_data, backend="PYT", mode=what_is_second_dim, method='opt', number_distractors=2, max_attempts=1000, max_iter=1000,silent=False)
    actual_model_label = unwrapped_model(torch.from_numpy(sample_to_explain['x']).to(torch.float32)).argmax(-1).item()

    inputx = torch.from_numpy(sample_to_explain['x']).to('cpu', torch.float32)
    # x1 = model(inputx)
    # x2 = model.model(inputx)

    exp = explainer_method.explain(sample_to_explain['x'], orig_class=actual_model_label, target=1-actual_model_label,
                                   max_n_feats=cf_feats_to_switch, tau_confidence=tau_confidence)

    plot_original_overlap_counterfactual(sample_to_explain['x'], exp, feature_names, explanation_output_folder)
    return exp


def do_NUNCF(explanation_config, sample_to_explain, explanation_output_folder):
    method_name = "nuncf"
    model = explanation_config["model"]
    feature_names = explanation_config["feature_names"]
    what_is_second_dim = explanation_config["what_is_second_dim"]
    background_data = explanation_config["background_data"]
    background_data['x'] = background_data['x'][:explanation_config[method_name]['num_background_used']]
    background_data['y'] = background_data['y'][:explanation_config[method_name]['num_background_used']]
    unwrapped_model = model.model
    unwrapped_model.eval()
    comte_formatted_data = tuple([background_data['x'], background_data['y'].squeeze()])
    #mt = 'dtw_bary_center'
    #mt = 'NUN_CF'
    mt = 'NG'

    dist = "dtw"
    #dist = "euclidean"

    class nuncf_wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)

    explainer_method = NativeGuideCF(nuncf_wrapper(unwrapped_model.to('cpu')), comte_formatted_data, backend="PYT",
                                     mode=what_is_second_dim, method=mt, distance_measure=dist, n_neighbors=1,
                                     max_iter=3000)

    actual_model_label = model.model(torch.from_numpy(sample_to_explain['x']).to(torch.float32)).argmax(-1).item()
    exp = explainer_method.explain(x=sample_to_explain['x'], y=actual_model_label)

    #For NG
    #exp = (exp[0].transpose(0,2,1), exp[1])
    # For NUN_CF
    exp = (exp[0].reshape(1, 48, 17), exp[1])

    plot_original_overlap_counterfactual(sample_to_explain['x'], exp, feature_names, explanation_output_folder)
    return exp


def do_Anchors(explanation_config, sample_to_explain, explanation_output_folder):
    method_name = "anchors"
    model = explanation_config["model"]
    feature_names = explanation_config["feature_names"]
    background_data = explanation_config["background_data"]
    # background_data['x'] = background_data['x'][:explanation_config[method_name]['num_background_used']]
    # background_data['y'] = background_data['y'][:explanation_config[method_name]['num_background_used']]
    config = explanation_config["experiment_config"]
    tau_setting = explanation_config[method_name]["tau"]
    delta_setting = explanation_config[method_name]["delta"]
    #tau = 0.1, stop_on_first = True, batch_size = 50, delta = 0.05

    one_locs = np.argwhere(background_data['y'] == 1)[:,0]
    zero_locs = np.argwhere(background_data['y'] == 0)[:,0]
    background_data_xs = np.concatenate((one_locs[:50], zero_locs[:50]))


    flat_background = background_data['x'][background_data_xs].reshape([-1] + [np.prod(background_data['x'].shape[1:])])
    feat_names = []
    cat_dict = {}
    for i in range(flat_background.shape[1]):
        flat_loc = i%sample_to_explain['x'].shape[-1]
        which_stack = i//sample_to_explain['x'].shape[-1]
        name = feature_names[flat_loc]

        flat_feat_name = name + f"_{which_stack}"
        feat_names.append(flat_feat_name)
        GLASGOW_COMASCALE_MAPPING = {
            'Glascow coma scale eye opening': {'Spontaneously': 4, 'To Pressure': 2, 'To Sound': 3, 'No Response': 1},
            'Glascow coma scale motor response': {'Obeys Commands': 6, 'Localizing': 5, 'Normal Flexion': 4,
                                                  'Abnormal Flexion': 3, 'Extension': 2, 'No Response': 1},
            'Glascow coma scale verbal response': {'Oriented': 5, 'Confused': 4, 'Words': 3, 'Sounds': 2,
                                                   'No Response': 1}}
        if name in config['categorical_features']:
            #GLASGOW_COMASCALE_MAPPING[name]
            if name in GLASGOW_COMASCALE_MAPPING.keys():
                cat_dict[i] = {v:k for k,v in GLASGOW_COMASCALE_MAPPING[name].items()} #instead should be a dict of values

    explainer = anchor_tabular.AnchorTabularExplainer(
        ["Passed","Survived"],
        feat_names,
        flat_background.astype(np.float32),
        cat_dict
    )
    # exp = explainer.explain_instance(sample_to_explain['x'].flatten(), model.predict, threshold=0.9)#, desired_label=1-pred_label)#, tau=0.1, stop_on_first=True)
    exp = explainer.explain_instance(sample_to_explain['x'].flatten().astype(np.float32), model.predict, threshold=0.95,
                                     tau=tau_setting, stop_on_first=True, batch_size=10, delta=delta_setting)

    print(exp)
    print('Anchor: %s' % (' AND '.join(exp.names())))
    print('Precision: %.2f' % exp.precision())
    print('Coverage: %.2f' % exp.coverage())
    exp_str = (' AND '.join(exp.names()))
    return [exp, exp_str]


def do_Dynamask(explanation_config, sample_to_explain, explanation_output_folder):
    model = explanation_config["model"]
    feature_names = explanation_config["feature_names"]

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class dynamask_wrapper():
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            with torch.no_grad():
                return self.model(x.unsqueeze(0))
        def __call__(self, x):
            return self.forward(x)


    unwrapped_model = dynamask_wrapper(model)

    pert = GaussianBlur(device)
    mask = Mask(pert, device, task='classification')
    mask.fit(torch.from_numpy(sample_to_explain['x'].squeeze()).to(device, torch.float32), f=unwrapped_model,
             loss_function=mse, keep_ratio=0.2,
             time_reg_factor=0.95,
             size_reg_factor_init=0.1
             )  # Select the 10% most important features

    mask_tensor = mask.mask_tensor
    ids_time = None
    ids_feature = None
    submask_tensor_np = mask.extract_submask(mask_tensor, ids_time=ids_time, ids_feature=ids_feature).numpy()
    plot_original_line_with_vals(sample_to_explain, submask_tensor_np, feature_names, explanation_output_folder)

    return submask_tensor_np


def do_LORE(explanation_config, sample_to_explain, explanation_output_folder):
    method_name = "lore"
    background_data = explanation_config["background_data"]
    background_data['x'] = background_data['x'][:explanation_config[method_name]['num_background_used']]
    background_data['y'] = background_data['y'][:explanation_config[method_name]['num_background_used']]
    model = explanation_config["model"]
    model_type = explanation_config["model_type"]
    feature_names = explanation_config["feature_names"]
    window_length = explanation_config["window_length"]
    cur_device = explanation_config["experiment_config"]["device"]

    # from LASTS_explainer.lasts import Lasts
    # from LASTS_explainer.neighborhood_generators import NeighborhoodGenerator
    # from LASTS_explainer.variational_autoencoder import load_model
    # from LASTS_explainer.lasts_utils import choose_z
    # from LASTS_explainer.sbgdt import Sbgdt

    blackbox = model
    class encoder_wrapper():
        def __init__(self, model):
            super().__init__()
            self.model = model
        def predict(self, x):
            x = torch.from_numpy(x).to("cpu", torch.float32).reshape(x.shape[0], -1)
            return self.model.forward(x).detach()
    class decoder_wrapper():
        def __init__(self, model):
            super().__init__()
            self.model = model

        def predict(self, x):
            x = x.to(torch.float32)
            return self.model.forward(x).reshape(-1, 48, 17).detach()

    autoencoder = torch.load("_saved_models/autoencoderPresentationTesting_epoch27_loss596045.pt")
    encoder = encoder_wrapper(autoencoder.encoder.to("cpu"))
    decoder = decoder_wrapper(autoencoder.decoder.to("cpu"))

    random_state = 0
    z = choose_z(sample_to_explain['x'], encoder, decoder, n=1000, x_label=blackbox.predict(sample_to_explain['x'])[0], blackbox=blackbox, check_label=True)

    neighborhood_generator = NeighborhoodGenerator(blackbox, decoder)
    neigh_kwargs = {
        "balance": False,
        "n": 50,
        "n_search": 10000000,
        "threshold": 200,
        "sampling_kind": "uniform_sphere",
        "kind": "gaussian_matched",
        "verbose": False,
        "stopping_ratio": 0.01,
        "downward_only": True,
        "redo_search": True,
        "forced_balance_ratio": 0.5,
        "cut_radius": True
    }


    model.skip_autobatch = True
    lasts_ = Lasts(blackbox,
                   encoder,
                   decoder,
                   sample_to_explain['x'],
                   neighborhood_generator,
                   z_fixed=z,
                   labels=["survive", "death"]
                   )
    out = lasts_.generate_neighborhood(**neigh_kwargs)
    model.skip_autobatch = False
    surrogate = Sbgdt(shapelet_model_params={"max_iter": 50}, random_state=random_state)
    # surrogate = Saxdt(random_state=np.random.seed(0))
    # WARNING: you need a forked version of the library sktime in order to view SAX plots
    # SUBSEQUENCE EXPLAINER

    lasts_.fit_surrogate(surrogate, binarize_labels=True)
    # SUBSEQUENCE TREE
    lasts_.surrogate._graph
    # VARIOUS PLOTS
    lasts_.plot_binary_heatmap(step=5)
    lasts_.plot_factual_and_counterfactual()

    #############################################################

    #from xailib.explainers.lore_explainer import LoreTabularExplainer
    # lore_explainer = LOREM( K=background_data['x'],
    #                         bb_predict=model.predict,
    #                         feature_names=feature_names,
    #                         class_name=["Died", "Survived"],
    #                         class_values=[0, 1],
    #                         numeric_columns=feature_names,
    #                         features_map=None)
    #
    # exp = lore_explainer.explain_instance(x, samples=1000, use_weights=True)
    # temp = LoreTabularExplanation(exp)
    # print(temp)

    ######################################################################


    # if sample_to_explain['x'].shape[1] % window_length != 0:
    #     raise NotImplementedError(f"FATAL - time series length {sample_to_explain['x'].shape[1]} must be divisible by the window size {window_length}")
    #
    # gtw = StationaryWindowSHAP(model, window_length, B_ts=background_data['x'], test_ts=sample_to_explain['x'], model_type=model_type)
    #
    # gtw.explainer = shap.KernelExplainer(gtw.wraper_predict, gtw.background_data)
    # shap_values = gtw.explainer.shap_values(gtw.test_data)
    # shap_values = np.array(shap_values)
    #
    # sv = np.repeat(shap_values.flatten().reshape((-1, gtw.num_window)), window_length, axis=1)
    #
    # plot_original_line_with_vals(sample_to_explain, sv.transpose(), feature_names, explanation_output_folder)
    # return sv


def do_AnchorsAlt(explanation_config, sample_to_explain, explanation_output_folder):
    method_name = "anchorsalt"
    model = explanation_config["model"]
    feature_names = explanation_config["feature_names"]
    background_data = explanation_config["background_data"]
    # background_data['x'] = background_data['x'][:explanation_config[method_name]['num_background_used']]
    # background_data['y'] = background_data['y'][:explanation_config[method_name]['num_background_used']]
    config = explanation_config["experiment_config"]
    tau_setting = explanation_config[method_name]["tau"]
    delta_setting = explanation_config[method_name]["delta"]
    #tau = 0.1, stop_on_first = True, batch_size = 50, delta = 0.05

    one_locs = np.argwhere(background_data['y'] == 1)[:,0]
    zero_locs = np.argwhere(background_data['y'] == 0)[:,0]
    background_data_xs = np.concatenate((one_locs[:50], zero_locs[:50]))


    flat_background = background_data['x'][background_data_xs].reshape([-1] + [np.prod(background_data['x'].shape[1:])])
    feat_names = []
    cat_dict = {}
    for i in range(flat_background.shape[1]):
        flat_loc = i%sample_to_explain['x'].shape[-1]
        which_stack = i//sample_to_explain['x'].shape[-1]
        name = feature_names[flat_loc]

        flat_feat_name = name + f"_{which_stack}"
        feat_names.append(flat_feat_name)
        GLASGOW_COMASCALE_MAPPING = {
            'Glascow coma scale eye opening': {'Spontaneously': 4, 'To Pressure': 2, 'To Sound': 3, 'No Response': 1},
            'Glascow coma scale motor response': {'Obeys Commands': 6, 'Localizing': 5, 'Normal Flexion': 4,
                                                  'Abnormal Flexion': 3, 'Extension': 2, 'No Response': 1},
            'Glascow coma scale verbal response': {'Oriented': 5, 'Confused': 4, 'Words': 3, 'Sounds': 2,
                                                   'No Response': 1}}
        if name in config['categorical_features']:
            if name in GLASGOW_COMASCALE_MAPPING.keys():
                cat_dict[i] = {v:k for k,v in GLASGOW_COMASCALE_MAPPING[name].items()} #instead should be a dict of values



    def override_fn(full_sample):
        for single_feature_ts in full_sample:
            single_feat_explainer = anchor_tabular.AnchorTabularExplainer(
                ["Passed", "Survived"],
                feat_names,
                flat_background.astype(np.float32),
                cat_dict
            )

        exp = explainer.explain_instance(sample_to_explain['x'].flatten().astype(np.float32), model.predict, threshold=0.95,
                                     tau=tau_setting, stop_on_first=True, batch_size=5, delta=delta_setting)

    for i in range(5):
        thread = threading.Thread(
            target=override_fn,
            args=(f'computer_{i}',),
        )
        thread.start()

    print(exp)
    print('Anchor: %s' % (' AND '.join(exp.names())))
    print('Precision: %.2f' % exp.precision())
    print('Coverage: %.2f' % exp.coverage())

    return exp.exp_map