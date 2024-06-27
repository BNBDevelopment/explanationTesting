import os
import pickle

import matplotlib
import numpy as np
import seaborn
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import hist, bar


def process_counterfactual(batch, **kwargs):
    raw_data = []
    raw_idxs = []
    raw_lbls = []
    for x in batch:
        cf = x[0][0]
        cf_label = x[0][1].item()
        loc = x[1]

        raw_data.append(cf)
        raw_idxs.append(loc)
        raw_lbls.append(cf_label)

    preprocess = kwargs['preprocess']
    counterfactuals = np.stack(raw_data).squeeze()
    if preprocess:
        counterfactuals = preprocess(counterfactuals)

    orig_inputs = kwargs['orig_inputs'][raw_idxs]
    model = kwargs['model'].cpu()


    saved_metrics = {}
    # metrics:
    orig_confidence = model(torch.from_numpy(orig_inputs).to(torch.float32))
    counterf_confidence = model(torch.from_numpy(counterfactuals).to(torch.float32))

    diff_model_confidence = orig_confidence - counterf_confidence
    batch_avg_diff_confidence = np.mean(diff_model_confidence.detach().numpy(), axis=0)
    batch_std_diff_confidence = np.std(diff_model_confidence.detach().numpy(), axis=0)

    batch_og_std_for_timepoint_for_feature = np.std(orig_inputs, axis=0)
    # batch_og_mean_std_along_timepoints = np.mean(np.std(orig_inputs, axis=2), axis=0)
    # batch_og_mean_std_along_features = np.mean(np.std(orig_inputs, axis=1), axis=0)

    batch_cf_std_for_timepoint_for_feature = np.std(counterfactuals, axis=0)
    # batch_cf_mean_std_along_timepoints = np.mean(np.std(counterfactuals, axis=1), axis=0)
    # batch_cf_mean_std_along_features = np.mean(np.std(counterfactuals, axis=1), axis=0)

    diff_between_cf_and_orig = orig_inputs - counterfactuals
    batch_mean_diff_along_timepoints = np.mean(diff_between_cf_and_orig, axis=1)
    batch_mean_diff_along_features = np.mean(diff_between_cf_and_orig, axis=2)

    batch_median_diff_along_timepoints = np.median(diff_between_cf_and_orig, axis=1)
    batch_median_diff_along_features = np.median(diff_between_cf_and_orig, axis=2)

    saved_metrics['batch_avg_diff_confidence'] = batch_avg_diff_confidence
    saved_metrics['batch_std_diff_confidence'] = batch_std_diff_confidence
    saved_metrics['batch_og_std_for_timepoint_for_feature'] = batch_og_std_for_timepoint_for_feature
    saved_metrics['batch_cf_std_for_timepoint_for_feature'] = batch_cf_std_for_timepoint_for_feature
    saved_metrics['batch_mean_diff_along_timepoints'] = batch_mean_diff_along_timepoints
    saved_metrics['batch_mean_diff_along_features'] = batch_mean_diff_along_features
    saved_metrics['batch_median_diff_along_timepoints'] = batch_median_diff_along_timepoints
    saved_metrics['batch_median_diff_along_features'] = batch_median_diff_along_features
    saved_metrics['og_mean'] = np.mean(orig_inputs, axis=0)
    saved_metrics['cf_mean'] = np.mean(counterfactuals, axis=0)

    return saved_metrics


def process_featureattribution(batch, **kwargs):
    saved_metrics = {}
    shap_vals = []
    raw_idxs = []
    for x in batch:
        cf = x[0]
        loc = x[1]

        shap_vals.append(cf)
        raw_idxs.append(loc)
    orig_inputs = kwargs['orig_inputs'][raw_idxs]
    model = kwargs['model'].cpu()
    shap_vals = np.stack(shap_vals).squeeze()

    # metrics:
    orig_confidence = model(torch.from_numpy(orig_inputs).to(torch.float32))
    batch_avg_confidence = np.mean(orig_confidence.detach().numpy(), axis=0)
    batch_std_confidence = np.std(orig_confidence.detach().numpy(), axis=0)

    batch_og_std_for_timepoint_for_feature = np.std(orig_inputs, axis=0)

    batch_mean_along_timepoints = np.mean(np.mean(shap_vals, axis=0), axis=1)
    batch_mean_along_features = np.mean(np.mean(shap_vals, axis=0), axis=0)
    batch_avg_std_along_timepoints = np.mean(np.std(shap_vals, axis=2), axis=0)
    batch_avg_std_along_features = np.mean(np.std(shap_vals, axis=1), axis=0)

    batch_median_along_timepoints = np.mean(np.median(shap_vals, axis=0), axis=1)
    batch_median_along_features = np.mean(np.median(shap_vals, axis=0), axis=0)

    saved_metrics['batch_avg_confidence'] = batch_avg_confidence
    saved_metrics['batch_std_confidence'] = batch_std_confidence
    saved_metrics['batch_og_std_for_timepoint_for_feature'] = batch_og_std_for_timepoint_for_feature
    saved_metrics['batch_mean_along_timepoints'] = batch_mean_along_timepoints
    saved_metrics['batch_mean_along_features'] = batch_mean_along_features
    saved_metrics['batch_avg_std_along_timepoints'] = batch_avg_std_along_timepoints
    saved_metrics['batch_avg_std_along_features'] = batch_avg_std_along_features
    saved_metrics['batch_median_along_timepoints'] = batch_median_along_timepoints
    saved_metrics['batch_median_along_features'] = batch_median_along_features

    return saved_metrics

def process_batch_comte(batch, **kwargs):
    return process_counterfactual(batch, **kwargs)

def process_batch_gradcam(batch, **kwargs):
    return process_featureattribution(batch, **kwargs)

def process_batch_nuncaf(batch, **kwargs):
    return process_counterfactual(batch, **kwargs)


def process_batch_timeshap(batch, **kwargs):
    saved_metrics = {}
    feature_shap_vals = []
    timepoint_shap_vals = []
    raw_idxs = []
    for x in batch:
        cf = x[0]
        loc = x[1]

        feature_shap_vals.append(cf[1].to_numpy()[:,3])
        timepoint_shap_vals.append(cf[0].to_numpy()[:,3])
        raw_idxs.append(loc)
    orig_inputs = kwargs['orig_inputs'][raw_idxs]
    model = kwargs['model'].cpu()
    #shap_vals = np.stack(shap_vals)

    # metrics:
    orig_confidence = model(torch.from_numpy(orig_inputs).to(torch.float32))
    batch_avg_confidence = np.mean(orig_confidence.detach().numpy(), axis=0)
    batch_std_confidence = np.std(orig_confidence.detach().numpy(), axis=0)

    batch_og_std_for_timepoint_for_feature = np.std(orig_inputs, axis=0)

    batch_mean_along_timepoints = np.mean(timepoint_shap_vals, axis=0)
    batch_mean_along_features = np.mean(feature_shap_vals, axis=0)

    try:
        batch_avg_std_along_timepoints = np.std(np.stack(timepoint_shap_vals), axis=0)
    except  TypeError as e:
        batch_avg_std_along_timepoints = 0
    try:
        batch_avg_std_along_features = np.std(feature_shap_vals, axis=0)
    except TypeError as e:
        batch_avg_std_along_features = 0

    batch_median_along_timepoints = np.median(timepoint_shap_vals, axis=0)
    batch_median_along_features = np.median(feature_shap_vals, axis=0)

    saved_metrics['batch_avg_confidence'] = batch_avg_confidence
    saved_metrics['batch_std_confidence'] = batch_std_confidence
    saved_metrics['batch_og_std_for_timepoint_for_feature'] = batch_og_std_for_timepoint_for_feature
    saved_metrics['batch_mean_along_timepoints'] = batch_mean_along_timepoints
    saved_metrics['batch_mean_along_features'] = batch_mean_along_features
    saved_metrics['batch_avg_std_along_timepoints'] = batch_avg_std_along_timepoints
    saved_metrics['batch_avg_std_along_features'] = batch_avg_std_along_features
    saved_metrics['batch_median_along_timepoints'] = batch_median_along_timepoints
    saved_metrics['batch_median_along_features'] = batch_median_along_features

    return saved_metrics


def process_batch_windowshap(batch, **kwargs):
    return process_featureattribution(batch, **kwargs)


def reshape_to_examples(batch):
    return np.reshape(batch, (-1, 70, 15))

PROCESSES = {'comte': [process_batch_comte],
             'gradcam': [process_batch_gradcam],
             'nuncaf': [process_batch_nuncaf],
             'time_shap': [process_batch_timeshap],
             'window_shap': [process_batch_windowshap] }

PREPROCESSES = {'comte': [None],
             'gradcam': [None],
             'nuncaf': [reshape_to_examples],
             'time_shap': [None],
             'window_shap': [None] }


def load_analysis_files(directory, item_categories=[]):
    results_dict = {x: [] for x in item_categories}

    for cat in item_categories:
        list_of_pickels = os.listdir(directory + cat)
        #list_of_pickels.sort()
        #analysis_items = {x:[] for x in item_categories}

        for foldername in list_of_pickels:
            temp_path = directory + cat + "/" + foldername
            temp_files = os.listdir(temp_path)
            for file in temp_files:
                if file[-4:] == ".pkl":
                    temp_f = open(temp_path + "/" + file, "rb")
                    res_objects = pickle.load(temp_f)
                    temp_f.close()
                    results_dict[cat].append(res_objects)
    return results_dict


def process_perBatch_results(per_batch_data):
    final_res = {}
    for catg, batches_results in per_batch_data.items():
        keys_to_avg = batches_results[0].keys()
        avg_dict = {x:None for x in keys_to_avg}
        for item in batches_results:
            for metric_name, metric_val in item.items():
                if avg_dict[metric_name] is None:
                    avg_dict[metric_name] = metric_val/len(batches_results)
                else:
                    avg_dict[metric_name] += metric_val/len(batches_results)
        final_res[catg] = avg_dict

    return final_res

def plot_original_overlap_counterfactual(test_item, explan_res, feature_names, explanation_output_folder, image_name_prefix="", n_plots_horiz=3):
    n_plots_vert = (len(feature_names) // n_plots_horiz) + 1
    figure, axis = plt.subplots(n_plots_vert, n_plots_horiz, figsize=(10, 10), layout='constrained')
    ts_len = explan_res[0].shape[1]
    time_tick_lbls = [""] * ts_len
    for j, _ in enumerate(time_tick_lbls):
        if j % 5 == 0:
            time_tick_lbls[j] = str(j)
    for i in range(len(feature_names)):
        axis[i // n_plots_horiz, i % n_plots_horiz].plot(list(range(0, ts_len)), explan_res[0][:, :, i].flatten(), color='r', label='Counterfactual')
        axis[i // n_plots_horiz, i % n_plots_horiz].plot(list(range(0, ts_len)), test_item[:, :, i].flatten(), color='b', label='Original Instance')
        axis[i // n_plots_horiz, i % n_plots_horiz].set_title(f"{feature_names[i]}")
        axis[i // n_plots_horiz, i % n_plots_horiz].set_xticks(list(range(0, ts_len)), labels=time_tick_lbls)
        axis[i // n_plots_horiz, i % n_plots_horiz].set_ylabel("Original Instance", color='b')

        overlay_plot = axis[i // n_plots_horiz, i % n_plots_horiz].twinx()
        overlay_plot.set_ylabel("Counterfactual", color='r')

    figure.savefig(f"{explanation_output_folder}{image_name_prefix}allFeatures.png")
    plt.close()


def plot_original_line_with_vals(test_item, explan_res, feature_names, explanation_output_folder, image_name_prefix="", n_plots_horiz=3):
    n_plots_vert = (len(feature_names) // n_plots_horiz) + 1
    figure, axis = plt.subplots(n_plots_vert, n_plots_horiz, figsize=(10, 10), layout='constrained')
    ts_len = explan_res.shape[0]
    time_tick_lbls = [""] * ts_len
    for j, _ in enumerate(time_tick_lbls):
        if j % 5 == 0:
            time_tick_lbls[j] = str(j)
    for i in range(len(feature_names)):
        feature_name = feature_names[i]
        working_subplot = axis[i // n_plots_horiz, i % n_plots_horiz]
        working_subplot.set_title(f"{feature_name}")
        working_subplot.set_xticks(list(range(0, ts_len)), labels=time_tick_lbls)

        working_subplot.plot(list(range(0, ts_len)), test_item['x'][:, :, i].flatten(), color='b', label='Original Instance')
        working_subplot.set_ylabel("Original Instance", color='b')
        working_subplot.tick_params(axis='y', colors='b')

        overlay_plot = working_subplot.twinx()
        overlay_plot.bar(list(range(0, ts_len)), explan_res[:, i].flatten(), color='r', label='Importance Values')
        overlay_plot.set_ylabel("Importance Values", color='r')
        overlay_plot.tick_params(axis='y', colors='r')

        working_subplot.set_zorder(overlay_plot.get_zorder() + 1)
        working_subplot.patch.set_visible(False)

        #overlay_plot.set_title(f"{feature_name}")
        #overlay_plot.set_xticks(list(range(0, ts_len)), labels=time_tick_lbls)
    figure.savefig(f"{explanation_output_folder}{image_name_prefix}allFeatures.png")
    plt.close()



def run_analysis():
    directory = '_saved_models/LSTM_mine/'
    item_categories = ['Anchors', 'COMTE', 'GradCAM', 'NUNCF', 'WindowSHAP']
    replication_size = 10

    data = load_analysis_files(directory, item_categories)

    ctg_batch_res = {x:[] for x in item_categories}
    for catg in item_categories:
        catg_data = data[catg]

        for i in range(0, len(catg_data)//replication_size):
            start = i * replication_size
            end = (i+1) * replication_size
            same_sample_batch = catg_data[start:end]

            proc_fn = PROCESSES[catg][0]
            pre_fn = PREPROCESSES[catg][0]
            batch_results = proc_fn(same_sample_batch, model=model, orig_inputs=orig_inputs, preprocess=pre_fn)
            ctg_batch_res[catg].append(batch_results)

    all_res = process_perBatch_results(ctg_batch_res)


def cf_histogram(exp_names, float_tolerance=0.0):
    data = {}
    for exp_name in exp_names:
        foldpath = f"_saved_models/{exp_name}/"
        filepath = foldpath + "all_explanations.pkl"
        f = open(filepath, "rb")
        all_explanations = pickle.load(f)
        f.close()
        exp_results = all_explanations['CoMTE']['result_store']
        comte_exps = exp_results['explanations']
        comte_orig = exp_results['samples_explained']

        exp_feat_n = exp_name.split("_feat")[1][0]

        for i, orig in enumerate(comte_orig):
            orig_x = orig['x']
            exp_x = comte_exps[i][0]

            if exp_feat_n in data.keys():
                data[exp_feat_n].append((orig_x, exp_x))
            else:
                data[exp_feat_n] = [(orig_x, exp_x)]

    histogram_data = {}

    for feat_n, list_tups in data.items():
        for tup in list_tups:
            itm = tup[0]
            exp = tup[1]
            diff = np.sum(exp - itm, axis=1).squeeze()
            diff_loc = (diff > float_tolerance) + (diff < -float_tolerance)
            sum_diffs = sum(diff_loc)

            if sum_diffs in histogram_data.keys():
                histogram_data[sum_diffs] += 1
            else:
                histogram_data[sum_diffs] = 1
        #histogram_data[feat_n] = len(list_tups)

    print(f"Histogram Data: {histogram_data}")
    t_l = list(range(1, len(exp_names)+1))
    h = bar(height=list(histogram_data.values()), x=t_l, tick_label=t_l )
    for i in range(0,5):
        plt.text(i+1, list(histogram_data.values())[i], list(histogram_data.values())[i])
    #plt.text(4, list(histogram_data.values())[3], list(histogram_data.values())[3], ha='center')

    temp = {int(k):len(v) for k,v in data.items()}
    print(f"Raw Data Counts: {temp}")
    print("Finished")

if __name__ == "__main__":
    #run_analysis()
    #e_list = ["Experiment_COMTE_MAX_1","Experiment_COMTE_MAX_2","Experiment_COMTE_MAX_3","Experiment_COMTE_MAX_4"]

    e_list = ["Exp_COMTE_conf99_feat1","Exp_COMTE_conf99_feat2","Exp_COMTE_conf99_feat3","Exp_COMTE_conf99_feat4","Exp_COMTE_conf99_feat5"]
    #e_list = ["Exp_COMTE_conf999_feat1","Exp_COMTE_conf999_feat2","Exp_COMTE_conf999_feat3","Exp_COMTE_conf999_feat4","Exp_COMTE_conf999_feat5"]
    #e_list = ["Exp_COMTE_conf656_feat1","Exp_COMTE_conf656_feat2","Exp_COMTE_conf656_feat3","Exp_COMTE_conf656_feat4","Exp_COMTE_conf656_feat5"]
    #e_list = ["Exp_COMTE_conf935_feat1", "Exp_COMTE_conf935_feat2","Exp_COMTE_conf935_feat3","Exp_COMTE_conf935_feat4","Exp_COMTE_conf935_feat5"]
    cf_histogram(e_list)