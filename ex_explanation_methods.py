import copy

import pandas as pd
import torch

from ForkAnchors.anchor import anchor_image
from ForkDynamask.attribution.mask import Mask
from ForkDynamask.attribution.perturbation import GaussianBlur
from ForkDynamask.utils.losses import mse
# from TSInterpret.InterpretabilityModels.counterfactual.COMTECF import COMTECF
# from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF import NativeGuideCF
# from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR

from _required_Packages.ForkAnchors.anchor import anchor_tabular
from _required_Packages.ForkTSInterpret.TSInterpret.InterpretabilityModels.Saliency.TSR import TSR
from _required_Packages.ForkTSInterpret.TSInterpret.InterpretabilityModels.counterfactual.COMTECF import COMTECF
from _required_Packages.ForkTSInterpret.TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF import \
    NativeGuideCF
from _required_Packages.ForkWindowSHAP.windowshap import StationaryWindowSHAP

from analysis import plot_original_overlap_counterfactual, plot_original_line_with_vals
import numpy as np
import shap
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# def do_comte(model, train_x, train_y, test_x, test_y, test_id_to_explain=0):
#     # will not work, requires a deprecated package (MLRose) that depends on a deprecated scikit-learn version (sklearn, no longer available)
#
#
#     test_y = test_y.to_frame().astype(int)
#     test_x.index.name = "node_id"
#     test_y.columns = ["label"]
#
#     names = ('node_id', 'timestamp')
#     mi_lbl = [[i for i in range(test_y.shape[0])], [0 for i in range(test_y.shape[0])]]
#     test_y.index = pd.MultiIndex.from_arrays(mi_lbl, names=names)
#
#     model.eval()
#     wrapped_model = modelwrapper(model)
#     pipeline = Pipeline([
#         ('clf', wrapped_model)
#     ], verbose=True)
#
#     preds = wrapped_model.predict(test_x.to_numpy())
#
#     print("F1 score:", f1_score(test_y, preds, average='weighted'))
#     for label, i in zip(np.unique(test_y), f1_score(test_y, preds, average=None)):
#         print("\t", label, i)
#
#
#     label_list = np.unique(test_y)
#     cf = confusion_matrix(test_y, preds, labels=label_list).astype(float)
#     for i in range(len(cf)):
#         cf[i] = [x / cf[i].sum() for x in cf[i]]
#     sns.heatmap(cf, annot=True, xticklabels=label_list, yticklabels=label_list)
#     plt.show()
#
#     train_y = train_y.to_frame().astype(int)
#     train_x.index.name = "node_id"
#     train_y.columns = ["label"]
#     names = ('node_id', 'timestamp')
#     mi_lbl = [[i for i in range(train_y.shape[0])], [0 for i in range(train_y.shape[0])]]
#     train_y.index = pd.MultiIndex.from_arrays(mi_lbl, names=names)
#     train_x.columns = [i for i in range(train_x.shape[1])]
#     test_x.columns = [i for i in range(test_x.shape[1])]
#
#     from explainers import OptimizedSearch
#     comte = OptimizedSearch(pipeline, train_x, train_y, silent=False, threads=1)
#     single_sample = pd.DataFrame(test_x.iloc[test_id_to_explain,:]).transpose()
#     single_sample_lbl = test_y.iloc[test_id_to_explain, :].item()
#     single_sample_pred = wrapped_model.predict(single_sample).item()
#     new_desired_label = 1 - single_sample_pred
#     #single_sample.columns = [i for i in range(single_sample.shape[1])]
#
#     explanation = comte.explain(single_sample, to_maximize=new_desired_label, savefig=False)
#     print('###########')
#     print(explanation)
#     print('###########')





#################################################################################


def do_WindowSHAP(explanation_config, sample_to_explain, explanation_output_folder):
    background_data = explanation_config["background_data"]
    background_data['x'] = background_data['x'][:100]
    background_data['y'] = background_data['y'][:100]
    model = explanation_config["model"]
    model_type = explanation_config["model_type"]
    feature_names = explanation_config["feature_names"]
    window_length = explanation_config["window_length"]

    if sample_to_explain['x'].shape[1] % window_length != 0:
        raise NotImplementedError(f"FATAL - time series length {sample_to_explain['x'].shape[1]} must be divisible by the window size {window_length}")

    gtw = StationaryWindowSHAP(model, window_length, B_ts=background_data['x'], test_ts=sample_to_explain['x'], model_type=model_type)

    gtw.explainer = shap.KernelExplainer(gtw.wraper_predict, gtw.background_data)
    shap_values = gtw.explainer.shap_values(gtw.test_data)
    shap_values = np.array(shap_values)

    sv = np.repeat(shap_values.flatten().reshape((-1, gtw.num_window)), window_length, axis=1)

    # for var_i, var_name in enumerate(feature_names):
    #     heat_map(start=0, stop=sample_to_explain['x'].shape[1], x=sample_to_explain['x'][:,:, var_i].flatten(), shap_values=sv[var_i].flatten(), var_name=var_name, plot_type='bar', image_save_path=explanation_output_folder)

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
    return exp




def do_COMTE(explanation_config, sample_to_explain, explanation_output_folder):
    model = explanation_config["model"]
    feature_names = explanation_config["feature_names"]
    what_is_second_dim = explanation_config["what_is_second_dim"]
    background_data = explanation_config["background_data"]
    background_data['x'] = background_data['x'][:100]
    background_data['y'] = background_data['y'][:100]

    unwrapped_model = model.model

    comte_formatted_data = tuple([background_data['x'], background_data['y'].squeeze()])
    explainer_method = COMTECF(unwrapped_model.to('cpu'), comte_formatted_data, backend="PYT", mode=what_is_second_dim, method='opt', number_distractors=2, max_attempts=1000, max_iter=1000,silent=False)
    actual_model_label = unwrapped_model(torch.from_numpy(sample_to_explain['x']).to(torch.float32)).argmax(-1).item()
    exp = explainer_method.explain(sample_to_explain['x'], orig_class=actual_model_label, target=1-actual_model_label)

    plot_original_overlap_counterfactual(sample_to_explain['x'], exp, feature_names, explanation_output_folder)
    return exp




def do_NUNCF(explanation_config, sample_to_explain, explanation_output_folder):
    model = explanation_config["model"]
    feature_names = explanation_config["feature_names"]
    what_is_second_dim = explanation_config["what_is_second_dim"]
    background_data = explanation_config["background_data"]
    # background_data['x'] = background_data['x'][:100]
    # background_data['y'] = background_data['y'][:100]
    unwrapped_model = model.model
    comte_formatted_data = tuple([background_data['x'], background_data['y'].squeeze()])
    mt = 'dtw_bary_center'

    class nuncf_wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)

    explainer_method = NativeGuideCF(nuncf_wrapper(unwrapped_model.to('cpu')), comte_formatted_data, backend="PYT", mode=what_is_second_dim, method=mt, distance_measure="dtw", n_neighbors=1, max_iter=500)

    exp = explainer_method.explain(x=sample_to_explain['x'], y=sample_to_explain['y'])
    plot_original_overlap_counterfactual(sample_to_explain['x'], exp, feature_names, explanation_output_folder)
    return exp


def do_Anchors(explanation_config, sample_to_explain, explanation_output_folder):
    model = explanation_config["model"]
    feature_names = explanation_config["feature_names"]
    background_data = explanation_config["background_data"]
    config = explanation_config["experiment_config"]

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
        if name in config['categorical_features']:
            cat_dict[i] = flat_feat_name

    explainer = anchor_tabular.AnchorTabularExplainer(
        ["Passed","Survived"],
        feat_names,
        flat_background.astype(np.float32),
        cat_dict
    )
    # exp = explainer.explain_instance(sample_to_explain['x'].flatten(), model.predict, threshold=0.9)#, desired_label=1-pred_label)#, tau=0.1, stop_on_first=True)
    exp = explainer.explain_instance(sample_to_explain['x'].flatten().astype(np.float32), model.predict, threshold=0.9, tau=0.9, stop_on_first=True, batch_size=1, delta=0.5)

    print(exp)
    print('Anchor: %s' % (' AND '.join(exp.names())))
    print('Precision: %.2f' % exp.precision())
    print('Coverage: %.2f' % exp.coverage())

    return exp.exp_map

def do_Dynamask(explanation_config, sample_to_explain, explanation_output_folder):
    # from attribution.mask import Mask
    # from attribution.perturbation import GaussianBlur
    # from utils.losses import mse
    model = explanation_config["model"]
    feature_names = explanation_config["feature_names"]
    background_data = explanation_config["background_data"]
    config = explanation_config["experiment_config"]

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class dynamask_wrapper():
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            with torch.no_grad():
                return self.model(x)
        def __call__(self, x):
            return self.forward(x)

    # Define a random input:
    # X = torch.randn(10, 3).to(device)  # The shape of the input has to be (T, N_features)
    unwrapped_model = dynamask_wrapper(model)
    # Fit a mask to the input with a Gaussian Blur perturbation:
    pert = GaussianBlur(device)
    mask = Mask(pert, device)
    mask.fit(torch.from_numpy(sample_to_explain['x'].squeeze()).to(device, torch.float32), unwrapped_model, loss_function=mse, keep_ratio=0.1,
             size_reg_factor_init=0.01)  # Select the 10% most important features

    # Plot the resulting saliency map:
    #mask.plot_mask()


    mask_tensor = mask.mask_tensor
    ids_time = None
    ids_feature = None
    submask_tensor_np = mask.extract_submask(mask_tensor, ids_time=ids_time, ids_feature=ids_feature).numpy()
    #df = pd.DataFrame(data=np.transpose(submask_tensor_np), index=ids_feature, columns=ids_time)
    # Generate heatmap plot
    #color_map = sns.diverging_palette(10, 133, as_cmap=True)
    #heat_map = sns.heatmap(data=df, cmap=color_map, cbar_kws={"label": "Mask"}, vmin=0, vmax=1)

    plot_original_line_with_vals(sample_to_explain, submask_tensor_np, feature_names, explanation_output_folder)



    # background_data = explanation_config["background_data"]
    # background_data['x'] = background_data['x'][:100]
    # background_data['y'] = background_data['y'][:100]
    # model = explanation_config["model"]
    # model_type = explanation_config["model_type"]
    # feature_names = explanation_config["feature_names"]
    # window_length = explanation_config["window_length"]
    #
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

    # for var_i, var_name in enumerate(feature_names):
    #     heat_map(start=0, stop=sample_to_explain['x'].shape[1], x=sample_to_explain['x'][:,:, var_i].flatten(), shap_values=sv[var_i].flatten(), var_name=var_name, plot_type='bar', image_save_path=explanation_output_folder)

    # plot_original_line_with_vals(sample_to_explain, sv.transpose(), feature_names, explanation_output_folder)
    # return sv
