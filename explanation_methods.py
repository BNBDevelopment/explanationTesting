import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

import data_loading
import utils
from util import heat_map


class modelwrapper():
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.column_names = ["x1"]
        self.classes_ = np.array([x for x in range(model.n_classes)])

    def predict(self, x):
        res = self.predict_proba(x)
        pred = np.argmax(res, axis=-1)
        return pred

    def predict_proba(self, x):
        if issubclass(x.__class__, pd.DataFrame):
            x = x.to_numpy()
        x = torch.from_numpy(x).to(self.model.fc1.bias.get_device()).to(torch.float32)
        res = self.model(x)
        return res.detach().cpu().numpy()


#test_pd_x, test_pd_y, _, _ = gen_multivar_regression_casual_data(num_fake_samples=10, num_features=N_FEATS, causal_y_idxs=CIDS)

#################################################################################
#MACE:
    #MACE does not appear to work for multivariate timeseries data

# def model_wrapper(x: Timeseries):
#     temp_x = torch.from_numpy(x.data)
#     out = model(temp_x.to(model.l1.bias.get_device()).to(torch.float32))
#     return torch.argmax(out).detach().item()
#
# explainer = MACEExplainer(
#     training_data=Timeseries.from_pd(train_pd_x),
#     predict_function=model_wrapper,
#     mode="forecasting",
#     threshold=0.001
# )
#
# test_x = Timeseries.from_pd(test_pd_x.iloc[0:3,:])
# explanations = explainer.explain(test_x)
# explanations.ipython_plot()


#################################################################################
#LASTS: not functional

# from lasts.blackboxes.loader import cached_blackbox_loader
# from lasts.datasets.datasets import build_cbf
# from lasts.autoencoders.variational_autoencoder import load_model
# from lasts.utils import get_project_root, choose_z
# from lasts.surrogates.shapelet_tree import ShapeletTree
# from lasts.neighgen.counter_generator import CounterGenerator
# from lasts.wrappers import DecoderWrapper
# from lasts.surrogates.utils import generate_n_shapelets_per_size
# from lasts.explainers.lasts import Lasts
# import numpy as np
#
# random_state = 0
# np.random.seed(random_state)
# dataset_name = "cbf"
#
# _, _, _, _, _, _, X_exp_train, y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test = build_cbf(n_samples=600, random_state=random_state)
#
# blackbox = cached_blackbox_loader("cbf_knn.joblib")
# encoder, decoder, autoencoder = load_model("/mnt/d/research/craven/baselines/explanationTesting/lasts/autoencoders/cached/vae/cbf/cbf_vae")
#
# i = 0
# x = X_exp_test[i].ravel().reshape(1, -1, 1)
# z_fixed = choose_z(x, encoder, decoder, n=1000, x_label=blackbox.predict(x)[0], blackbox=blackbox, check_label=True, mse=False)
#
# neighgen = CounterGenerator(blackbox, DecoderWrapper(decoder), n_search=10000)
#
# n_shapelets_per_size = generate_n_shapelets_per_size(X_exp_train.shape[1])
# surrogate = ShapeletTree(random_state=random_state, shapelet_model_kwargs={...})
#
# lasts_ = Lasts(blackbox, encoder, DecoderWrapper(decoder), neighgen, surrogate, verbose=True, binarize_surrogate_labels=True, labels=["cylinder", "bell", "funnel"])
#
# lasts_.fit(x, z_fixed)
#
# exp = lasts_.explain()
#
# lasts_.plot("latent_space")
# lasts_.plot("morphing_matrix")
# lasts_.plot("counterexemplar_interpolation")
# lasts_.plot("manifest_space")
# lasts_.plot("saliency_map")
# lasts_.plot("subsequences_heatmap")
# lasts_.plot("rules")
# lasts_.neighgen.plotter.plot_counterexemplar_shape_change()



#################################################################################
#CoMTE:
def do_comte(model, train_x, train_y, test_x, test_y):
    # will not work, requires a deprecated package (MLRose) that depends on a deprecated scikit-learn version (sklearn, no longer available)


    from sklearn.pipeline import Pipeline
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

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
    single_sample = pd.DataFrame(test_x.iloc[0,:]).transpose()
    single_sample_lbl = test_y.iloc[0, :].item()
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

def do_WindowSHAP(model, train_pd_x, test_pd_x):
    import numpy as np
    from windowshap import  StationaryWindowSHAP
    import timeit
    import shap

    train_x = train_pd_x.to_numpy()
    test_x = test_pd_x.to_numpy()

    train_x = np.expand_dims(train_x, -1)
    test_x = np.expand_dims(test_x, -1)

    num_background = 50
    num_test = 28
    background_data, test_data = train_x[:num_background], test_x[num_test:num_test+2]


    wrapped_model = modelwrapper(model)

    tic = timeit.default_timer()
    ts_phi_1 = np.zeros((len(test_data),test_data.shape[1], test_data.shape[2]))
    for i in range(len(test_data)):
        window_len = 16
        gtw = StationaryWindowSHAP(wrapped_model, window_len, B_ts=background_data, test_ts=test_data[i:i+1], model_type='lstm')

        gtw.explainer = shap.KernelExplainer(gtw.wraper_predict, gtw.background_data)
        shap_values = gtw.explainer.shap_values(gtw.test_data)
        shap_values = np.array(shap_values)

        #stretch shapleys over windows
        #n_repeats = test_data.shape[1] // window_len
        sv = np.repeat(shap_values.flatten(), window_len, axis=0)

        heat_map(start=0, stop=test_data.shape[1], x=test_data[i:i+1].flatten(), shap_values=sv, var_name='Observed', plot_type='bar')

        print("finally working!!!")

        #ts_phi_1[i,:,:] = gtw.shap_values()
    print('Total time: ' + str(timeit.default_timer()-tic))



    var = 0
    phi_index = 0
    heat_map(start=0, stop=120, x=test_x[num_test + phi_index, :, var], shap_values=ts_phi_1[phi_index, :, var], var_name='Observed', plot_type='bar')
    # heat_map(start=0, stop=120, x=test_x[num_test + phi_index, :, var], shap_values=ts_phi_2[phi_index, :, var], var_name='Observed', plot_type='bar')
    # heat_map(start=0, stop=120, x=test_x[num_test + phi_index, :, var], shap_values=ts_phi_3[phi_index, :, var], var_name='Variable', plot_type='bar')