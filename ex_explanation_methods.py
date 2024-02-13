import pathlib
import random

import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from TSInterpret.InterpretabilityModels.counterfactual.COMTECF import COMTECF
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR
from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF import NativeGuideCF

#from timeshap.explainer import calc_local_report, local_event, local_pruning, local_feat, TimeShapKernel
from util import heat_map
import numpy as np
from windowshap import StationaryWindowSHAP
import timeit
import shap
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


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


    if issubclass(train_x.__class__, np.ndarray):
        train_x = train_x.pd

    test_y = test_y.to_frame().astype(int)
    num_idx = 1
    #ts_to_explain.index.name = ["x"+str(i) for i in range(0, num_idx)]\
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


    # from explainers import OptimizedSearch
    # comte = OptimizedSearch(pipeline, test_x, test_y, silent=False, threads=1, num_distractors=2)
    #
    # true_select = 0 #if normal
    # pred_select = 1 #but predicted abnormal
    #
    # indices_test = []
    # for idx, (true, pred) in enumerate(zip(test_y.to_numpy(), preds)):
    #     if true.item() == true_select and pred.item() == pred_select:
    #         indices_test.append(idx)
    #
    # for ind in indices_test:
    #     x_test = test_x.loc[[ind], :]
    #     explanation = comte.explain(x_test,to_maximize=0,savefig=False)
    #     print('###########')
    #     print(explanation)
    #     print('###########')
    #     break



#################################################################################
#WindowSHAP:

class brits_wrapper():
    def __init__(self, model):
        self.model = model
    def predict(self, x):
        res = self.model.predict(x)
        return res['imputation']

def do_WindowSHAP(model, config, train_pd_x, test_pd_x, window_len=1, feature_names = [], wrap_model=True, model_type='lstm', num_background=50, test_idx=28, num_test_samples=1):
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
    background_data = train_x[background_start:background_start+num_background]
    test_data = test_x[test_idx:test_idx + num_test_samples]

    if wrap_model:
        wrapped_model = modelwrapper(model, n_classes=config['num_classes'])
    else:
        wrapped_model = brits_wrapper(model)

    tic = timeit.default_timer()
    ts_phi_1 = np.zeros((len(test_data),test_data.shape[1], test_data.shape[2]))
    for i in range(len(test_data)):
        gtw = StationaryWindowSHAP(wrapped_model, window_len, B_ts=background_data, test_ts=test_data[i:i+1], model_type=model_type)

        gtw.explainer = shap.KernelExplainer(gtw.wraper_predict, gtw.background_data)
        shap_values = gtw.explainer.shap_values(gtw.test_data)
        shap_values = np.array(shap_values)

        #stretch shapleys over windows
        #n_repeats = test_data.shape[1] // window_len

        #sv = np.repeat(shap_values.flatten(), window_len, axis=0)
        sv = np.repeat(shap_values.flatten().reshape((-1, gtw.num_window)), window_len, axis=1)
        saved_svs.append(sv)

        for var_i, var_name in enumerate(feature_names):
            #heat_map(start=0, stop=test_data.shape[1], x=test_data[i:i+1].flatten(), shap_values=sv, var_name='Observed', plot_type='bar')
            heat_map(start=0, stop=test_data.shape[1], x=test_data[i:i + 1, :, var_i].flatten(),
                     shap_values=sv[var_i].flatten(), var_name=var_name, plot_type='bar', image_save_path=img_save_path)

        print("finally working!!!")
    return saved_svs
        #ts_phi_1[i,:,:] = gtw.shap_values()



    var = 0
    phi_index = 0
    heat_map(start=0, stop=120, x=test_x[test_idx + phi_index, :, var], shap_values=ts_phi_1[phi_index, :, var], var_name='Observed', plot_type='bar')
    # heat_map(start=0, stop=120, x=test_x[num_test + phi_index, :, var], shap_values=ts_phi_2[phi_index, :, var], var_name='Observed', plot_type='bar')
    # heat_map(start=0, stop=120, x=test_x[num_test + phi_index, :, var], shap_values=ts_phi_3[phi_index, :, var], var_name='Variable', plot_type='bar')



def do_GradCAM(model, configuration, train_x, test_x, test_y, feature_names, wrap_model, model_type, num_background, test_idx,
               num_test_samples):
    from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR

    #eval_approach = 'IG'
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

    explainer_method.plot(np.array([test_x[test_idx, :, :]]), exp, figsize=(12.8, 9.6), heatmap=False, save=img_save_path + f"item_{test_idx}_feature_gradcam_result.png", feature_names=feature_names)
    explainer_method.plot(np.array([np.sum(test_x[test_idx, :, :], axis=1)]), exp, figsize=(12.8, 9.6), heatmap=False, save=img_save_path + f"item_{test_idx}_timepointSum_gradcam_result.png")

    print("Finished gradcam")
    return exp

def plot_original_overlap_counterfactual(test_item, explan_res, feature_names, test_idx):
    figure, axis = plt.subplots(8, 2, figsize=(10, 80), layout='constrained')
    lbls = [""] * 70
    for j,_ in enumerate(lbls):
        if j%5 == 0:
            lbls[j] = str(j)
    for i in range(15):
        axis[i//2, i%2].plot(list(range(0, 70)), explan_res[0][:, :, i].flatten(), color='r', label='counterfactual')
        axis[i//2, i%2].plot(list(range(0, 70)), test_item[:, :, i].flatten(), color='b', label='original')
        axis[i//2, i%2].set_title(f"Feature {feature_names[i]}")
        axis[i//2, i%2].set_xticks(list(range(0,70)),  labels=lbls)
    figure.savefig(f"example_overlap_item_{test_idx}.png")


def do_COMTE(model, configuration, train_x, test_x, test_y, feature_names, wrap_model, model_type, num_background, test_idx,
               num_test_samples):
    data = (test_x, test_y)


    #eval_approach = 'IG'
    eval_approach = 'GRAD'
    cur_device = configuration['device']
    what_is_second_dim = 'time'

    explainer_method = COMTECF(model.to('cpu'), data, backend="PYT", mode=what_is_second_dim, method='opt', number_distractors=2, max_attempts=1000, max_iter=1000,silent=False)

    test_item = np.array([test_x[test_idx, :, :]])
    #test_labl = int(test_y[test_idx])
    actual_model_label = model(torch.from_numpy(test_item).to(torch.float32)).argmax(-1).item()

    exp = explainer_method.explain(test_item, orig_class=actual_model_label, target=1-actual_model_label)

    img_save_path = configuration['save_model_path'] + configuration['model_name'] + "/comte/"
    path = pathlib.Path(img_save_path)
    path.mkdir(parents=True, exist_ok=True)

    explainer_method.plot(original=test_item, org_label=actual_model_label, exp=exp[0], exp_label=exp[1], figsize=(12.8, 9.6), save_fig=img_save_path + f"item_{test_idx}_result.png")

    differences = exp[0] - test_item



    plot_original_overlap_counterfactual(test_item, exp, feature_names, test_idx)

    print("Finished comte")
    return exp




def do_NUNCF(model, configuration, train_x, test_x, test_y, feature_names, wrap_model, model_type, num_background, test_idx,
               num_test_samples):
    data = (test_x, test_y)


    #eval_approach = 'IG'
    eval_approach = 'GRAD'
    cur_device = configuration['device']
    what_is_second_dim = 'time'

    #mt = 'NUN_CF'
    mt = 'dtw_bary_center'
    #mt = 'NG'

    # class wrapper(torch.nn.Module):
    #     def __init__(self, model):
    #         self.model = model
    #     def forward(self, in_x):


    explainer_method = NativeGuideCF(model.to('cpu'), data, backend="PYT", mode=what_is_second_dim, method=mt, distance_measure="dtw", n_neighbors=1, max_iter=500)

    test_item = np.array([test_x[test_idx, :, :]])
    test_labl = int(test_y[test_idx])

    exp = explainer_method.explain(x=test_item, y=test_labl)

    img_save_path = configuration['save_model_path'] + configuration['model_name'] + "/nuncf/"
    path = pathlib.Path(img_save_path)
    path.mkdir(parents=True, exist_ok=True)

    explainer_method.plot(original=test_item, org_label=test_labl, exp=exp[0], exp_label=exp[1], figsize=(12.8, 9.6), save_fig=img_save_path + f"item_{test_idx}_{mt}_result.png")

    print("Finished nuncf")
    return exp




# def do_TimeSHAP(model, config, train_pd_x, test_pd_x, window_len=10, feature_names = [], wrap_model=True, model_type='lstm', num_background=50, test_idx=28, num_test_samples=1):
#     if issubclass(train_pd_x.__class__, pd.DataFrame):
#         train_x = train_pd_x.to_numpy()
#         train_x = np.expand_dims(train_x, -1)
#     else:
#         train_x = train_pd_x
#
#     if issubclass(test_pd_x.__class__, pd.DataFrame):
#         test_x = test_pd_x.to_numpy()
#         test_x = np.expand_dims(test_x, -1)
#     else:
#         test_x = test_pd_x
#
#     saved_svs = []
#
#     img_save_path = config['save_model_path'] + config['model_name'] + "/windowshap/"
#     path = pathlib.Path(img_save_path)
#     path.mkdir(parents=True, exist_ok=True)
#
#     #background_start = random.randint(0, len(train_x) - num_background)
#     #background_data = train_x[background_start:background_start+num_background]
#     #test_data = test_x[test_idx:test_idx + num_test_samples]
#
#     # if wrap_model:
#     #     wrapped_model = modelwrapper(model, n_classes=config['num_classes'])
#     # else:
#     #     wrapped_model = brits_wrapper(model)
#
#     # tic = timeit.default_timer()
#     # ts_phi_1 = np.zeros((len(test_data),test_data.shape[1], test_data.shape[2]))
#
#
#
#
#
#
#     from timeshap.wrappers import TorchModelWrapper
#     model_wrapped = TorchModelWrapper(model, batch_budget=350000) #orig_budget = 750000
#     f_hs = lambda x, y=None: model_wrapped.predict_last_hs(x, y)
#
#     average_event = pd.DataFrame(np.expand_dims(np.mean(train_x, axis=(0,1)), axis=0), columns=feature_names)
#     #pos_x_data = test_data
#     positive_sequence_id = None
#     sequence_id_feat = None
#     verbose = True
#     model_features = feature_names
#     plot_feats = {k:k for k in feature_names}
#
#     #test_data_sample = np.expand_dims(test_x[7], axis=0)
#     test_data_sample = np.expand_dims(test_x[test_idx], axis=0)
#
#     #explainer = TimeShapKernel(f_hs, average_event, 0, "feature")
#     #shap_values = explainer.shap_values(test_data_sample, pruning_idx=0, **{'nsamples': 1000})
#     #temp = np.array([f_hs(x) for x in test_x]).squeeze().argmax(-1)
#
#
#
#     pruning_dict = {'tol': -1.0, }
#     #pruning_dict = {'tol': 0.025, }
#     coal_plot_data, coal_prun_idx = local_pruning(f_hs, test_data_sample,pruning_dict, average_event,  verbose=verbose)
#     # coal_prun_idx is in negative terms
#     pruning_idx = test_data_sample.shape[1] + coal_prun_idx
#     # pruning_plot = plot_temp_coalition_pruning(coal_plot_data, coal_prun_idx, plot_limit=40)
#     # pruning_plot
#
#     event_dict = {'rs': 42, 'nsamples': 12000}
#     event_data = local_event(f_hs, test_data_sample, event_dict, positive_sequence_id, sequence_id_feat, average_event, pruning_idx)
#     # event_plot = plot_event_heatmap(event_data)
#     # event_plot
#
#     feature_dict = {'rs': 42, 'nsamples': 12000, 'feature_names': model_features, 'plot_features': plot_feats}
#     feature_data = local_feat(f_hs, test_data_sample, feature_dict, positive_sequence_id, sequence_id_feat, average_event, pruning_idx)
#     # feature_plot = plot_feat_barplot(feature_data, feature_dict.get('top_feats'), feature_dict.get('plot_features'))
#     # feature_plot
#
#
#     print("fin")
#     saved_svs.append(event_data)
#     saved_svs.append(feature_data)
#     saved_svs.append(coal_plot_data)
#
#
#
#
#     # for i in range(len(test_data)):
#     #     print("finally working!!!")
#     return saved_svs
#         #ts_phi_1[i,:,:] = gtw.shap_values()
#
#     # var = 0
#     # phi_index = 0
#     # heat_map(start=0, stop=120, x=test_x[test_idx + phi_index, :, var], shap_values=ts_phi_1[phi_index, :, var], var_name='Observed', plot_type='bar')
#
#     # heat_map(start=0, stop=120, x=test_x[num_test + phi_index, :, var], shap_values=ts_phi_2[phi_index, :, var], var_name='Observed', plot_type='bar')
#     # heat_map(start=0, stop=120, x=test_x[num_test + phi_index, :, var], shap_values=ts_phi_3[phi_index, :, var], var_name='Variable', plot_type='bar')