import pathlib
import random

import pandas as pd
import torch
from TSInterpret.InterpretabilityModels.counterfactual.COMTECF import COMTECF
from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF import NativeGuideCF
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR

import ex_dataset
from analysis import plot_original_overlap_counterfactual, plot_original_line_with_vals
from util import heat_map
import numpy as np
from windowshap import StationaryWindowSHAP
import shap
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from anchor import anchor_tabular


class modelwrapper():
    def __init__(self, model, n_classes=None, model_flag=None):
        super().__init__()
        self.model = model
        self.column_names = ["x1"]
        if n_classes is None:
            self.classes_ = np.array([x for x in range(model.n_classes)])
        else:
            self.classes_ = n_classes
        self.model_flag = model_flag

    def predict(self, x):
        res = self.predict_proba(x)
        pred = np.argmax(res, axis=-1)
        return pred

    def predict_proba(self, x):
        if self.model_flag == "brits":
            res = self.model.predict({"X":x})
            return res['imputation']
        else:
            if issubclass(x.__class__, pd.DataFrame):
                x = x.to_numpy()
            x = torch.from_numpy(x).to(self.model.fc.bias.get_device()).to(torch.float32)

            if x.size(0) > 128:
                print("NOTICE: Batching for explanation methods activated...")
                x_arr = torch.split(x, x.size(0)//(x.size(0)//64), dim=0)
                reses = []
                for x in x_arr:
                    res = self.model(x)
                    reses.append(res.detach().cpu().numpy())
                res = np.concatenate(reses, axis=0)
            else:
                res = self.model(x)
                res = res.detach().cpu().numpy()
            return res


def do_comte(model, train_x, train_y, test_x, test_y, test_id_to_explain=0):
    # will not work, requires a deprecated package (MLRose) that depends on a deprecated scikit-learn version (sklearn, no longer available)


    test_y = test_y.to_frame().astype(int)
    test_x.index.name = "node_id"
    test_y.columns = ["label"]

    names = ('node_id', 'timestamp')
    mi_lbl = [[i for i in range(test_y.shape[0])], [0 for i in range(test_y.shape[0])]]
    test_y.index = pd.MultiIndex.from_arrays(mi_lbl, names=names)

    model.eval()
    wrapped_model = modelwrapper(model)
    pipeline = Pipeline([
        ('clf', wrapped_model)
    ], verbose=True)

    preds = wrapped_model.predict(test_x.to_numpy())

    print("F1 score:", f1_score(test_y, preds, average='weighted'))
    for label, i in zip(np.unique(test_y), f1_score(test_y, preds, average=None)):
        print("\t", label, i)


    label_list = np.unique(test_y)
    cf = confusion_matrix(test_y, preds, labels=label_list).astype(float)
    for i in range(len(cf)):
        cf[i] = [x / cf[i].sum() for x in cf[i]]
    sns.heatmap(cf, annot=True, xticklabels=label_list, yticklabels=label_list)
    plt.show()

    train_y = train_y.to_frame().astype(int)
    train_x.index.name = "node_id"
    train_y.columns = ["label"]
    names = ('node_id', 'timestamp')
    mi_lbl = [[i for i in range(train_y.shape[0])], [0 for i in range(train_y.shape[0])]]
    train_y.index = pd.MultiIndex.from_arrays(mi_lbl, names=names)
    train_x.columns = [i for i in range(train_x.shape[1])]
    test_x.columns = [i for i in range(test_x.shape[1])]

    from explainers import OptimizedSearch
    comte = OptimizedSearch(pipeline, train_x, train_y, silent=False, threads=1)
    single_sample = pd.DataFrame(test_x.iloc[test_id_to_explain,:]).transpose()
    single_sample_lbl = test_y.iloc[test_id_to_explain, :].item()
    single_sample_pred = wrapped_model.predict(single_sample).item()
    new_desired_label = 1 - single_sample_pred
    #single_sample.columns = [i for i in range(single_sample.shape[1])]

    explanation = comte.explain(single_sample, to_maximize=new_desired_label, savefig=False)
    print('###########')
    print(explanation)
    print('###########')





#################################################################################
#WindowSHAP:

class brits_wrapper():
    def __init__(self, model):
        self.model = model
    def predict(self, x):
        res = self.model.predict(x)
        return res['imputation']


def getConfig_WindowSHAP(config):
    return_dict = {}
    relevant_config_section = config['explanation_methods']['window_shap']

    return_dict['window_len'] = relevant_config_section['window_len']
    return_dict['wrap_model'] = relevant_config_section['wrap_model']
    return_dict['model_type'] = config['model_type']
    return_dict['num_background'] = relevant_config_section['num_background']
    return_dict['test_idx'] = relevant_config_section['test_idx']
    return_dict['num_test_samples'] = relevant_config_section['num_test_samples']

def do_WindowSHAP(model, config, train_x, test_x, window_len=1, feature_names = [], model_type='lstm', num_background=50, test_idx=28, num_test_samples=1):
    saved_svs = []
    getConfig_WindowSHAP

    img_save_path = config['save_model_path'] + config['model_name'] + "/windowshap/"
    path = pathlib.Path(img_save_path)
    path.mkdir(parents=True, exist_ok=True)

    background_start = random.randint(0, len(train_x) - num_background)
    background_data = train_x[background_start:background_start+num_background]
    test_data = test_x[test_idx:test_idx + num_test_samples]

    for i in range(len(test_data)):
        gtw = StationaryWindowSHAP(model, window_len, B_ts=background_data, test_ts=test_data[i:i+1], model_type=model_type)

        gtw.explainer = shap.KernelExplainer(gtw.wraper_predict, gtw.background_data)
        shap_values = gtw.explainer.shap_values(gtw.test_data)
        shap_values = np.array(shap_values)

        sv = np.repeat(shap_values.flatten().reshape((-1, gtw.num_window)), window_len, axis=1)
        saved_svs.append(sv)

        for var_i, var_name in enumerate(feature_names):
            heat_map(start=0, stop=test_data.shape[1], x=test_data[i:i + 1, :, var_i].flatten(),
                     shap_values=sv[var_i].flatten(), var_name=var_name, plot_type='bar', image_save_path=img_save_path)
    return saved_svs




def do_GradCAM(model, configuration, train_x, test_x, test_y, feature_names, wrap_model, model_type, num_background, test_idx,
               num_test_samples):

    eval_approach = 'GRAD'
    cur_device = configuration['device']
    what_is_second_dim = 'time'

    explainer_method = TSR(model, NumTimeSteps=train_x.shape[-2], NumFeatures =train_x.shape[-1], method=eval_approach, mode=what_is_second_dim, device=cur_device)

    test_item = np.array([test_x[test_idx, :, :]])
    test_labl = int(test_y[test_idx])

    exp = explainer_method.explain(test_item, labels=test_labl, TSR=True)

    img_save_path = configuration['save_model_path'] + configuration['model_name'] + "/gradcam/"
    path = pathlib.Path(img_save_path)
    path.mkdir(parents=True, exist_ok=True)

    explainer_method.plot(np.array([test_x[test_idx, :, :]]), exp, figsize=(12.8, 9.6), heatmap=False, save=img_save_path + f"item_{test_idx}_feature_gradcam_result.png")
    explainer_method.plot(np.array([np.sum(test_x[test_idx, :, :], axis=1)]), exp, figsize=(12.8, 9.6), heatmap=False, save=img_save_path + f"item_{test_idx}_timepointSum_gradcam_result.png")

    plot_original_line_with_vals(test_item, exp, feature_names, test_idx)
    return exp




def do_COMTE(model, configuration, train_x, test_x, test_y, feature_names, wrap_model, model_type, num_background, test_idx,
               num_test_samples):
    data = (test_x, test_y.squeeze())
    what_is_second_dim = 'time'

    explainer_method = COMTECF(model.to('cpu'), data, backend="PYT", mode=what_is_second_dim, method='opt', number_distractors=2, max_attempts=1000, max_iter=1000,silent=False)

    test_item = np.array([test_x[test_idx, :, :]])
    actual_model_label = model(torch.from_numpy(test_item).to(torch.float32)).argmax(-1).item()

    exp = explainer_method.explain(test_item, orig_class=actual_model_label, target=1-actual_model_label)

    img_save_path = configuration['save_model_path'] + configuration['model_name'] + "/comte/"
    path = pathlib.Path(img_save_path)
    path.mkdir(parents=True, exist_ok=True)

    explainer_method.plot(original=test_item, org_label=actual_model_label, exp=exp[0], exp_label=exp[1], figsize=(12.8, 9.6), save_fig=img_save_path + f"item_{test_idx}_result.png")

    plot_original_overlap_counterfactual(test_item, exp, feature_names, test_idx)

    return exp




def do_NUNCF(model, configuration, train_x, test_x, test_y, feature_names, wrap_model, model_type, num_background, test_idx,
               num_test_samples):
    data = (test_x, test_y)

    what_is_second_dim = 'time'

    mt = 'dtw_bary_center'

    class nuncf_wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)

    explainer_method = NativeGuideCF(nuncf_wrapper(model.to('cpu')), data, backend="PYT", mode=what_is_second_dim, method=mt, distance_measure="dtw", n_neighbors=1, max_iter=500)

    test_item = np.array([test_x[test_idx, :, :]])
    test_labl = int(test_y[test_idx])

    exp = explainer_method.explain(x=test_item, y=test_labl)

    img_save_path = configuration['save_model_path'] + configuration['model_name'] + "/nuncf/"
    path = pathlib.Path(img_save_path)
    path.mkdir(parents=True, exist_ok=True)

    explainer_method.plot(original=test_item, org_label=test_labl, exp=exp[0], exp_label=exp[1], figsize=(12.8, 9.6), save_fig=img_save_path + f"item_{test_idx}_{mt}_result.png")

    return exp


def do_Anchors(model, config, train_pd_x, test_pd_x, window_len=1, feature_names = [], wrap_model=True, model_type='lstm', num_background=50, test_idx=28, num_test_samples=1):
    if issubclass(train_pd_x.__class__, pd.DataFrame):
        train_x = train_pd_x.to_numpy()
        train_x = np.expand_dims(train_x, -1)
    else:
        train_x = train_pd_x

    if issubclass(test_pd_x.__class__, pd.DataFrame):
        test_x = test_pd_x.to_numpy()
        test_x = np.expand_dims(test_x, -1)
    else:
        test_x = test_pd_x

    saved_svs = []

    img_save_path = config['save_model_path'] + config['model_name'] + "/windowshap/"
    path = pathlib.Path(img_save_path)
    path.mkdir(parents=True, exist_ok=True)

    background_start = random.randint(0, len(train_x) - num_background)
    test_data = test_x[test_idx:test_idx + num_test_samples]

    for i in range(len(test_data)):
        flat_x = train_x.reshape([-1] + [np.prod(train_x.shape[1:])])
        ex_dataset.HARDCODED_MIMICIII_INITIAL_FEATURES
        feat_names = []
        cat_dict = {}
        for i in range(flat_x.shape[1]):
            flat_loc = i%train_x.shape[-1]
            which_stack = i//train_x.shape[-1]
            name = list(ex_dataset.HARDCODED_MIMICIII_INITIAL_FEATURES.keys())[flat_loc]

            flat_feat_name = name + f"_{which_stack}"
            feat_names.append(flat_feat_name)
            if name in config['categorical_features']:
                cat_dict[i] = flat_feat_name

        explainer = anchor_tabular.AnchorTabularExplainer(
            ["Passed","Survived"],
            feat_names,
            flat_x,
            cat_dict
        )

        test_item = np.expand_dims(test_data.flatten(),0)
        model = model.cpu()
        test_out = model(torch.from_numpy(test_data).to(torch.float32))
        pred_label = torch.argmax(test_out).item()
        exp = explainer.explain_instance(test_item, model.forward, threshold=0.95, desired_label=pred_label, tau=0.85, stop_on_first=True)

        print(exp)

        print('Anchor: %s' % (' AND '.join(exp.names())))
        print('Precision: %.2f' % exp.precision())
        print('Coverage: %.2f' % exp.coverage())

        for var_i, var_name in enumerate(feature_names):
            #heat_map(start=0, stop=test_data.shape[1], x=test_data[i:i+1].flatten(), shap_values=sv, var_name='Observed', plot_type='bar')
            heat_map(start=0, stop=test_data.shape[1], x=test_data[i:i + 1, :, var_i].flatten(),
                     shap_values=sv[var_i].flatten(), var_name=var_name, plot_type='bar', image_save_path=img_save_path)
    return saved_svs