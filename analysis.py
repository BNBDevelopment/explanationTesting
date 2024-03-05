import os
import pickle

import numpy as np
import seaborn
import torch
from matplotlib import pyplot as plt


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
    for cat in item_categories:
        list_of_pickels = os.listdir(directory + cat)
        #list_of_pickels.sort()
        #analysis_items = {x:[] for x in item_categories}

        for foldername in list_of_pickels:
            f_type = foldername.split("_")

            item_num = f_type[-1].split('.')[0].replace("Item", "")
            try:
                item_num = int(item_num)//10
            except:
                item_num = 0



        temp_path = directory + cat + "/" + foldername
        temp_files = os.listdir(temp_path)
        for file in temp_files:
            if file[-4:] == ".pkl":
                temp_f = open(temp_path + "/" + file, "rb")
                res_objects = pickle.load(temp_f)
                temp_f.close()

                if item_num >= len(res_objects):
                    last_res = res_objects[-1]
                else:
                    last_res = res_objects[item_num]

                analysis_items[f_type].append(tuple([last_res, item_num]))

    return analysis_items


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

def run_analysis(model, orig_inputs):
    directory = 'pickel_results/'
    item_categories = ['comte', 'gradcam', 'nuncaf', 'time_shap', 'window_shap' ]
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


    #####################################################################################
    ########### Draw Combined Fig ###############################
    #####################################################################################
    fig = plt.figure(figsize=(10, 5))
    wid = 0.4
    num_pts = 70

    def normalize(x):
        return (x-np.min(x))/(np.max(x)-np.min(x))

    ################################################################################################

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    t1 = normalize(all_res['gradcam']['batch_mean_along_timepoints'])
    t2 = normalize(all_res['time_shap']['batch_mean_along_timepoints'].astype(float))
    t3 = normalize(all_res['window_shap']['batch_mean_along_features'])

    # for i, col in enumerate(random_df.columns[1:]):
    #     # random_df.plot(kind='scatter', x=col, y='MEDV', ax=ax[i])
    #     if i <= 6:
    #         sns.regplot(x=random_df[col], y=random_df["MEDV"], ax=ax[0, i])
    #     else:
    #         sns.regplot(x=random_df[col], y=random_df["MEDV"], ax=ax[1, i - 7])

    seaborn.regplot(x=t1, y=t1, ax=ax[0, 0], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=70)})
    seaborn.regplot(x=t1, y=t2, ax=ax[0, 1], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=70)})
    seaborn.regplot(x=t1, y=t3, ax=ax[0, 2], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=70)})

    seaborn.regplot(x=t2, y=t1, ax=ax[1, 0], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=70)})
    seaborn.regplot(x=t2, y=t2, ax=ax[1, 1], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=70)})
    seaborn.regplot(x=t2, y=t3, ax=ax[1, 2], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=70)})

    seaborn.regplot(x=t3, y=t1, ax=ax[2, 0], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=70)})
    seaborn.regplot(x=t3, y=t2, ax=ax[2, 1], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=70)})
    seaborn.regplot(x=t3, y=t3, ax=ax[2, 2], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=70)})

   # ax[1, 6].axis('off')  # HIDES AXES ON LAST ROW AND COL

    ax[0][0].set_ylabel('GradCAM')
    ax[0][0].set_xlabel('GradCAM')

    ax[0][1].set_ylabel('TimeShap')
    ax[0][1].set_xlabel('GradCAM')

    ax[0][2].set_ylabel('WindowSHAP')
    ax[0][2].set_xlabel('GradCAM')

    ax[1][0].set_ylabel('GradCAM')
    ax[1][0].set_xlabel('TimeShap')

    ax[1][1].set_ylabel('TimeShap')
    ax[1][1].set_xlabel('TimeShap')

    ax[1][2].set_ylabel('WindowSHAP')
    ax[1][2].set_xlabel('TimeShap')

    ax[2][0].set_ylabel('GradCAM')
    ax[2][0].set_xlabel('WindowSHAP')

    ax[2][1].set_ylabel('TimeShap')
    ax[2][1].set_xlabel('WindowSHAP')

    ax[2][2].set_ylabel('GradCAM')
    ax[2][2].set_xlabel('WindowSHAP')


    fig.suptitle('Pair-wise comparison of predicted importance values')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    plt.show()
    plt.savefig("scatter.png")
    plt.clf()
    plt.close()

    ################################################################################################

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    t1 = normalize(all_res['gradcam']['batch_mean_along_features'])
    t2 = normalize(all_res['time_shap']['batch_mean_along_features'].astype(float))
    t3 = normalize(all_res['window_shap']['batch_mean_along_timepoints'])

    # for i, col in enumerate(random_df.columns[1:]):
    #     # random_df.plot(kind='scatter', x=col, y='MEDV', ax=ax[i])
    #     if i <= 6:
    #         sns.regplot(x=random_df[col], y=random_df["MEDV"], ax=ax[0, i])
    #     else:
    #         sns.regplot(x=random_df[col], y=random_df["MEDV"], ax=ax[1, i - 7])

    seaborn.regplot(x=t1, y=t1, ax=ax[0, 0], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=15)})
    seaborn.regplot(x=t1, y=t2, ax=ax[0, 1], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=15)})
    seaborn.regplot(x=t1, y=t3, ax=ax[0, 2], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=15)})

    seaborn.regplot(x=t2, y=t1, ax=ax[1, 0], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=15)})
    seaborn.regplot(x=t2, y=t2, ax=ax[1, 1], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=15)})
    seaborn.regplot(x=t2, y=t3, ax=ax[1, 2], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=15)})

    seaborn.regplot(x=t3, y=t1, ax=ax[2, 0], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=15)})
    seaborn.regplot(x=t3, y=t2, ax=ax[2, 1], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=15)})
    seaborn.regplot(x=t3, y=t3, ax=ax[2, 2], scatter_kws={"color": seaborn.color_palette(palette='hls', n_colors=15)})

    # ax[1, 6].axis('off')  # HIDES AXES ON LAST ROW AND COL

    ax[0][0].set_ylabel('GradCAM')
    ax[0][0].set_xlabel('GradCAM')

    ax[0][1].set_ylabel('TimeShap')
    ax[0][1].set_xlabel('GradCAM')

    ax[0][2].set_ylabel('WindowSHAP')
    ax[0][2].set_xlabel('GradCAM')

    ax[1][0].set_ylabel('GradCAM')
    ax[1][0].set_xlabel('TimeShap')

    ax[1][1].set_ylabel('TimeShap')
    ax[1][1].set_xlabel('TimeShap')

    ax[1][2].set_ylabel('WindowSHAP')
    ax[1][2].set_xlabel('TimeShap')

    ax[2][0].set_ylabel('GradCAM')
    ax[2][0].set_xlabel('WindowSHAP')

    ax[2][1].set_ylabel('TimeShap')
    ax[2][1].set_xlabel('WindowSHAP')

    ax[2][2].set_ylabel('GradCAM')
    ax[2][2].set_xlabel('WindowSHAP')

    fig.suptitle('Pair-wise comparison of predicted importance values')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    plt.show()
    plt.savefig("scatter_mean_long_feat.png")
    plt.clf()
    plt.close()

    ################################################################################################
    p1 = normalize(all_res['gradcam']['batch_mean_along_features'])
    p2 = normalize(all_res['time_shap']['batch_mean_along_features'])
    p3 = normalize(all_res['window_shap']['batch_mean_along_timepoints'])


    p4 = normalize(all_res['gradcam']['batch_mean_along_timepoints'])
    p5 = normalize(all_res['time_shap']['batch_mean_along_timepoints'])
    p6 = normalize(all_res['window_shap']['batch_mean_along_features'])
    for x in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        print(f"Count Feature GradCAM for {x} is {(len(np.where(p1 >= x)[0])-1)/15}")
        print(f"Count Feature time_shap for {x} is {(len(np.where(p2 >= x)[0])-1)/15}")
        print(f"Count Feature window_shap for {x} is {(len(np.where(p3 >= x)[0])-1)/15}")
        print(f"Count Timepoint GradCAM for {x} is {(len(np.where(p4 >= x)[0])-1)/70}")
        print(f"Count Timepoint time_shap for {x} is {(len(np.where(p5 >= x)[0])-1)/70}")
        print(f"Count Timepoint window_shap for {x} is {(len(np.where(p6 >= x)[0])-1)/70}")

    print("exit")



    ################################################################################################

    t01 = all_res['comte']['batch_og_std_for_timepoint_for_feature']
    t02 = all_res['comte']['batch_cf_std_for_timepoint_for_feature']
    t03 = all_res['comte']['batch_mean_diff_along_timepoints']
    t04 = all_res['comte']['batch_mean_diff_along_features']

    t05 = all_res['nuncaf']['batch_og_std_for_timepoint_for_feature']
    t06 = all_res['nuncaf']['batch_cf_std_for_timepoint_for_feature']
    t07 = all_res['nuncaf']['batch_mean_diff_along_timepoints']
    t08 = all_res['nuncaf']['batch_mean_diff_along_features']
    np.mean(all_res['nuncaf']['og_mean'])
    np.mean(all_res['nuncaf']['cf_mean'])

    np.mean(all_res['gradcam']['batch_mean_along_timepoints'])
    np.mean(all_res['time_shap']['batch_mean_along_features'])
    np.mean(all_res['window_shap']['batch_mean_along_features'])

    x1 = [x - wid for x in range(1, num_pts+1)]
    y1 = normalize(all_res['gradcam']['batch_mean_along_timepoints']) #'batch_mean_along_features' 'batch_avg_std_along_timepoints' 'batch_avg_std_along_features'
    x2 = [x for x in range(1, num_pts+1)]
    y2 = normalize(all_res['time_shap']['batch_mean_along_timepoints'])
    x3 = [x + wid for x in range(1, num_pts+1)]
    y3 = normalize(all_res['window_shap']['batch_mean_along_features'])

    plt.bar(x1, y1, color='red', width=0.4, label='grad_cam')
    plt.bar(x2, y2, color='green', width=0.4, label='time_shap')
    plt.bar(x3, y3, color='blue', width=0.4, label='window_shap')

    labels = ['grad_cam', 'time_shap', 'window_shap']
    colors = {'grad_cam':'red', 'time_shap':'green', 'window_shap':'blue'}
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.xlabel("Timestep")
    plt.ylabel("Importance Value")
    plt.title("")
    plt.show()
    plt.savefig("norm_combined_by_timepoint.png")

    #####################################################################################
    ########### Draw Combined Fig ###############################
    #####################################################################################
    fig = plt.figure(figsize=(10, 5))
    wid = 0.4
    num_pts = 15

    x1 = [x - wid for x in range(1, num_pts+1)]
    y1 = normalize(all_res['gradcam']['batch_mean_along_features']) #'batch_mean_along_features' 'batch_avg_std_along_timepoints' 'batch_avg_std_along_features'
    x2 = [x for x in range(1, num_pts+1)]
    y2 = normalize(all_res['time_shap']['batch_mean_along_features'])
    x3 = [x + wid for x in range(1, num_pts+1)]
    y3 = normalize(all_res['window_shap']['batch_mean_along_timepoints'])

    plt.bar(x1, y1, color='red', width=0.4, label='grad_cam')
    plt.bar(x2, y2, color='green', width=0.4, label='time_shap')
    plt.bar(x3, y3, color='blue', width=0.4, label='window_shap')

    labels = ['grad_cam', 'time_shap', 'window_shap']
    colors = {'grad_cam':'red', 'time_shap':'green', 'window_shap':'blue'}
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.xlabel("Timestep")
    plt.ylabel("Importance Value")
    plt.title("")
    plt.show()
    plt.savefig("norm_combined_by_feature.png")


    #####################################################################################
    ########### Draw Combined Fig ###############################
    #####################################################################################
    # fig = plt.figure(figsize=(10, 5))
    # wid = 0.4
    # num_pts = 15
    #
    # x1 = [x - wid for x in range(1, num_pts+1)]
    # y1 = all_res['gradcam']['batch_mean_along_features'] #'batch_mean_along_features' 'batch_avg_std_along_timepoints' 'batch_avg_std_along_features'
    # x2 = [x for x in range(1, num_pts+1)]
    # y2 = all_res['time_shap']['batch_mean_along_features']
    # x3 = [x + wid for x in range(1, num_pts+1)]
    # y3 = all_res['window_shap']['batch_mean_along_timepoints']
    #
    # plt.bar(x1, y1, color='red', width=0.4, label='grad_cam')
    # plt.bar(x2, y2, color='green', width=0.4, label='time_shap')
    # plt.bar(x3, y3, color='blue', width=0.4, label='window_shap')
    #
    # labels = ['grad_cam', 'time_shap', 'window_shap']
    # colors = {'grad_cam':'red', 'time_shap':'green', 'window_shap':'blue'}
    # handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    # plt.legend(handles, labels)
    # plt.xlabel("Timestep")
    # plt.ylabel("Importance Value")
    # plt.title("")
    # plt.show()
    # plt.savefig("combined_by_feature.png")

def plot_original_overlap_counterfactual(test_item, explan_res, feature_names, explanation_output_folder, image_name_prefix="", n_plots_horiz=3):
    n_plots_vert = (len(feature_names) // n_plots_horiz) + 1
    figure, axis = plt.subplots(n_plots_vert, n_plots_horiz, figsize=(10, 10), layout='constrained')
    ts_len = explan_res[0].shape[1]
    time_tick_lbls = [""] * ts_len
    for j, _ in enumerate(time_tick_lbls):
        if j % 5 == 0:
            time_tick_lbls[j] = str(j)
    for i in range(len(feature_names)):
        axis[i // n_plots_horiz, i % n_plots_horiz].plot(list(range(0, ts_len)), explan_res[0][:, :, i].flatten(), color='r', label='counterfactual')
        axis[i // n_plots_horiz, i % n_plots_horiz].plot(list(range(0, ts_len)), test_item[:, :, i].flatten(), color='b', label='original')
        axis[i // n_plots_horiz, i % n_plots_horiz].set_title(f"{feature_names[i]}")
        axis[i // n_plots_horiz, i % n_plots_horiz].set_xticks(list(range(0, ts_len)), labels=time_tick_lbls)
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
        axis[i // n_plots_horiz, i % n_plots_horiz].bar(list(range(0, ts_len)), explan_res[:, i].flatten(), color='r', label='Counterfactual')
        axis[i // n_plots_horiz, i % n_plots_horiz].plot(list(range(0, ts_len)), test_item['x'][:, :, i].flatten(), color='b', label='original')
        axis[i // n_plots_horiz, i % n_plots_horiz].set_title(f"{feature_name}")
        axis[i // n_plots_horiz, i % n_plots_horiz].set_xticks(list(range(0, ts_len)), labels=time_tick_lbls)
    figure.savefig(f"{explanation_output_folder}{image_name_prefix}allFeatures.png")
    plt.close()



if __name__ == "__main__":
    directory = '_saved_models/LSTM_mine/'
    item_categories = ['Anchors', 'COMTE', 'GradCAM' , 'NUNCF', 'WindowSHAP']
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