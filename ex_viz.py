import pickle
import os

import numpy as np

from analysis import plot_original_line_with_vals, plot_original_overlap_counterfactual

temp_path = "_saved_models/LSTM_mine/all_explanations.pkl"
temp_f = open(temp_path, "rb")
res_objects = pickle.load(temp_f)
temp_f.close()

time_data = {k: {} for k in res_objects.keys()}
most_immportant_feats = {k: [] for k in res_objects.keys()}
for name, v in res_objects.items():
    if not name in ["GradCAM", "NUNCF"]:
        times = v["result_store"]["time_taken"]
        time_data[name]["mean"] = np.mean(times)
        time_data[name]["median"] = np.median(times)
        time_data[name]["min"] = np.min(times)
        time_data[name]["max"] = np.max(times)

        num_viz = 3
        originals = v["result_store"]["samples_explained"]
        explanations = v["result_store"]["explanations"]
        fts = ['Capillary refill rate', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']


        for i in range(num_viz):
            sample_to_explain = originals[i]
            importances = explanations[i]
            if name in ["WindowSHAP","GradCAM"]:
                if name == "WindowSHAP":
                    importances = importances.transpose()
                plot_original_line_with_vals(sample_to_explain, importances, feature_names=fts, explanation_output_folder="analysis")
                feature_importance_order = np.flip(np.argsort(np.sum(importances, axis=0)))
                most_immportant_feats[name].append(feature_importance_order)

            elif name in ["CoMTE", "NUNCF"]:
                plot_original_overlap_counterfactual(sample_to_explain['x'], importances, feature_names=fts, explanation_output_folder="analysis")
                #changed_feats = np.argmax(np.sum(importances))
                diff = np.sum(np.abs(sample_to_explain['x'] - importances[0]).squeeze(), axis=0)
                diff_idxs = np.flip(np.argsort(diff))
                # list_diff_feats_name = np.array(fts)[diff_idxs]
                # list_diff_feats = [list_diff_feats_name.tolist().index(x) for x in list_diff_feats_name]
                most_immportant_feats[name].append(diff_idxs)

            elif name in ["Anchors"]:
                print('Anchor: %s' % (' AND '.join(importances.names())))
                print('Precision: %.2f' % importances.precision())
                print('Coverage: %.2f' % importances.coverage())

print("Fin")




