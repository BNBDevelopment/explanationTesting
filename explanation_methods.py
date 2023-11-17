import torch

from util import heat_map


class modelwrapper():
    def __init__(self, model):
        super().__init__()
        self.model = model

    def predict(self, x):
        x = torch.from_numpy(x).to(self.model.l1.bias.get_device()).to(torch.float32)
        res = self.model(x)
        pred = torch.argmax(res, dim=-1)
        return pred.detach().cpu().numpy()


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
def do_comte(model, test_x, ts_to_explain):
    # will not work, requires a deprecated package (MLRose) that depends on a deprecated scikit-learn version (sklearn, no longer available)


    import logging
    from pathlib import Path
    import sys
    import data_loading
    import utils
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    from utils import TSFeatureGenerator


    timeseries, labels, test_timeseries, test_labels = data_loading.get_dataset('natops', binary=False);
    extractor = TSFeatureGenerator(threads=1, trim=0)

    # pipeline = Pipeline([
    #     ('assert1', utils.CheckFeatures()),
    #     ('features', utils.TSFeatureGenerator(threads=1, trim=0)),
    #     ('assert2', utils.CheckFeatures()),
    #     ('scaler', MinMaxScaler(feature_range=(-1, 1))),
    #     ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced'))
    # ], verbose=True)
    #pipeline.fit(timeseries, labels)
    #preds = pipeline.predict(test_timeseries)

    wrapped_model = modelwrapper(model)
    preds = wrapped_model.predict(test_x.to_numpy())

    print("F1 score:", f1_score(ts_to_explain, preds, average='weighted'))
    for label, i in zip(ts_to_explain.unique(), f1_score(ts_to_explain, preds, average=None)):
        print("\t", label, i)

    label_list = list(ts_to_explain.unique())
    cf = confusion_matrix(ts_to_explain, preds, labels=label_list).astype(float)
    for i in range(len(cf)):
        cf[i] = [x / cf[i].sum() for x in cf[i]]
    sns.heatmap(cf, annot=True, xticklabels=label_list, yticklabels=label_list)
    plt.show()

    import explainers
    from explainers import OptimizedSearch
    comte = OptimizedSearch(wrapped_model, ts_to_explain, labels, silent=False, threads=1)
    comte.explain(test_timeseries.loc[['5c15428439747d4a8fa8f85d_60'], :, :], to_maximize=6, savefig=False)

 

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