import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torcheval.metrics import BinaryAUROC
from torcheval.metrics.aggregation.auc import AUC
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import gru_ode_bayes
from gru_ode_bayes import data_utils
import tensorflow as tf


def format_pds(pd_x, pd_y, batch_sz, current_device):
    if issubclass(pd_x.__class__, pd.DataFrame):
        x = torch.tensor(pd_x.to_numpy()).to(current_device)
        y = torch.tensor(pd_y.to_numpy()).to(current_device)
    else:
        x = torch.tensor(pd_x).to(current_device)
        y = torch.tensor(pd_y).to(current_device)

    ds_set = torch.utils.data.TensorDataset(x, y)
    d_loader = torch.utils.data.DataLoader(ds_set, batch_size=batch_sz, shuffle=True, drop_last=True)

    return d_loader


def model_forward(model, loss_fn, x, y):
    # x = torch.tensor(pd_x.iloc[data_idx, :].to_numpy()).to(current_device)
    # y = torch.tensor(pd_y.iloc[data_idx, :].to_numpy()).to(current_device)
    # x.requires_grad = True
    # y.requires_grad = True
    x = x.to(torch.float32)
    y = y.squeeze().to(torch.long)

    output = model(x).to(torch.float32)

    # loss = loss_fn(output.squeeze(), y.squeeze())
    y = torch.nn.functional.one_hot(y, num_classes=2).to(torch.float32)
    loss = loss_fn(output, y)

    return loss, output

def train(model, config_dict, pd_x, pd_y, val_x=None, val_y=None):

    n_epochs = config_dict['n_epochs']
    lr = config_dict['lr']
    batch_sz = config_dict['batch_size']
    optimizer = config_dict['optimizer']
    current_device = config_dict['device']
    loss_fn = config_dict['loss_fn']

    if issubclass(model.__class__, BaseEstimator):
        model.fit(pd_x, pd_y)
    else:
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if current_device is None:
            current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"current_device: {current_device}")

        d_loader = format_pds(pd_x, pd_y, batch_sz, current_device)

        if not (val_x is None and val_y is None):
            val_loader = format_pds(val_x, val_y, batch_sz, current_device)

        model.train()
        model.to(torch.float32)
        model.to(current_device)

        best_aucroc = 0

        for epoch in range(1,n_epochs+1):
            print("---------------------------- START EPOCH ------------------------------")
            epoch_loss = 0
            e_iters = 0
            # for data_idx in range(0, len(pd_x)):
            for x, y in tqdm(d_loader, unit="batch"):
                e_iters += 1

                #run the model
                loss, outs = model_forward(model, loss_fn, x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().item()

            print(f"\nepoch loss: {epoch_loss/e_iters}")

            if val_x is None and val_y is None:
                print("Skipping validation...")
            else:
                print("Running validation...")
                metric_auc = AUC()
                metric_aucroc = BinaryAUROC()

                epoch_validation_loss = 0
                epoch_correct = 0
                epoch_incorrect = 0
                epoch_total = 0
                e_counts = 0

                false_pos = 0
                false_neg = 0
                true_pos = 0
                true_neg = 0

                for x, y in tqdm(val_loader, unit="batch"):
                    e_counts += 1

                    loss, outs = model_forward(model, loss_fn, x, y)

                    preds = torch.argmax(outs, dim=-1).detach()
                    incor = torch.sum(torch.abs(preds-y)).detach().item()
                    epoch_total += preds.size(0)
                    epoch_incorrect += incor
                    epoch_correct += preds.size(0) - incor

                    false_pos += sum(((preds == 1).to(torch.int8) + (y == 0).to(torch.int8)) == 2).detach().item()
                    false_neg += sum(((preds == 0).to(torch.int8) + (y == 1).to(torch.int8)) == 2).detach().item()
                    true_pos += sum(((preds == 1).to(torch.int8) + (y == 1).to(torch.int8)) == 2).detach().item()
                    true_neg += sum(((preds == 0).to(torch.int8) + (y == 0).to(torch.int8)) == 2).detach().item()

                    metric_auc.update(preds, y)
                    metric_aucroc.update(preds, y)

                    epoch_validation_loss += loss.detach().item()

                auc_val = metric_auc.compute()
                aucroc_val = metric_aucroc.compute()
                print(f"\nEpoch {epoch} \t\t Validation Loss: {epoch_validation_loss/e_counts} \t\t Total: {epoch_total} \t\t Correct: {epoch_correct} \t\t Fails: {epoch_incorrect}")
                print(f"Epoch {epoch} \t\t Accuracy: {epoch_correct/epoch_total}\t\t AUCROC: {aucroc_val.detach().item()} \t\t AUC: {auc_val.detach().item()}")
                print(f"Epoch {epoch} \t\t false_pos: {false_pos}\t\t false_neg: {false_neg}\t\t true_pos: {true_pos}\t\t true_neg: {true_neg}\t\t")
                print("---------------------------- END EPOCH ------------------------------")
                metric_auc.reset()
                if aucroc_val > best_aucroc:
                    best_aucroc = aucroc_val
        print(f"FINAL\t\t Best AUCROC: {best_aucroc}")
    return model


# def bayes_train(model, config_dict, pd_x, pd_y):
#
#     d_loader = format_pds(pd_x, pd_y, config_dict['batch_size'], config_dict['device'])
#     optimizer = config_dict['optimizer']
#     device = config_dict['device']
#     class_criterion = config_dict['loss_fn']
#
#     print("Start Training")
#     val_metric_prev = -1000
#     for epoch in range(config_dict['n_epochs']):
#         model.train()
#         total_train_loss = 0
#         auc_total_train = 0
#         tot_loglik_loss = 0
#         for i, b in enumerate(tqdm(d_loader)):
#
#             optimizer.zero_grad()
#             times = b["times"]
#             time_ptr = b["time_ptr"]
#             X = b["X"].to(device)
#             M = b["M"].to(device)
#             obs_idx = b["obs_idx"]
#             cov = b["cov"].to(device)
#             labels = b["y"].to(device)
#             batch_size = labels.size(0)
#
#             h0 = 0  # torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(device)
#             hT, loss, class_pred, mse_loss = model(times, time_ptr, X, M, obs_idx, delta_t=config_dict["delta_t"],
#                                                      T=config_dict["T"], cov=cov)
#
#             total_loss = (loss + config_dict["lambda"] * class_criterion(class_pred, labels)) / batch_size
#             total_train_loss += total_loss
#             tot_loglik_loss += mse_loss
#             try:
#                 auc_total_train += roc_auc_score(labels.detach().cpu(), torch.sigmoid(class_pred).detach().cpu())
#             except ValueError:
#                 if config_dict["verbose"] >= 3:
#                     print("Single CLASS ! AUC is erroneous")
#                 pass
#
#             total_loss.backward()
#             optimizer.step()
#
#         info = {'training_loss': total_train_loss.detach().cpu().numpy() / (i + 1),
#                 'AUC_training': auc_total_train / (i + 1), "loglik_loss": tot_loglik_loss.detach().cpu().numpy()}
#         print(f"NegLogLik Loss train : {tot_loglik_loss.detach().cpu().numpy()}")
#
#         data_utils.adjust_learning_rate(optimizer, epoch, config_dict["lr"])
#
#     print(f"Finished training GRU-ODE for Climate.")
#
#     return
#
# def tf_train(model, config_dict, pd_x, pd_y):
#     train_dropout_prob = config_dict["dropout"]
#     training_epochs = config_dict["n_epochs"]
#     optimizer = config_dict["optimizer"]
#
#     init = tf.global_variables_initializer()
#     saver = tf.train.Saver()
#
#     with tf.Session() as sess:
#         sess.run(init)
#         for epoch in range(training_epochs):  #
#             # Loop over all batches
#             total_cost = 0
#             for i in range(number_train_batches):  #
#                 # batch_xs is [number of patients x sequence length x input dimensionality]
#                 # batch_xs, batch_ys, batch_ts = data_train_batches[i], labels_train_batches[i], \
#                 #     elapsed_train_batches[i]
#                 #batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])
#                 sess.run(optimizer, feed_dict={model.input: batch_xs, model.labels: batch_ys, \
#                                                model.keep_prob: train_dropout_prob, model.time: batch_ts})
#
#         print("Training is over!")
#         #saver.save(sess, model_path)
#
#         Y_pred = []
#         Y_true = []
#         Labels = []
#         Logits = []
#         for i in range(number_train_batches):  #
#             batch_xs, batch_ys, batch_ts = data_train_batches[i], labels_train_batches[i], \
#                 elapsed_train_batches[i]
#             batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])
#             c_train, y_pred_train, y_train, logits_train, labels_train = sess.run(lstm.get_cost_acc(), feed_dict={
#                 lstm.input:
#                     batch_xs, lstm.labels: batch_ys, \
#                 lstm.keep_prob: train_dropout_prob, lstm.time: batch_ts})
#
#             if i > 0:
#                 Y_true = np.concatenate([Y_true, y_train], 0)
#                 Y_pred = np.concatenate([Y_pred, y_pred_train], 0)
#                 Labels = np.concatenate([Labels, labels_train], 0)
#                 Logits = np.concatenate([Logits, logits_train], 0)
#             else:
#                 Y_true = y_train
#                 Y_pred = y_pred_train
#                 Labels = labels_train
#                 Logits = logits_train
#
#         total_acc = accuracy_score(Y_true, Y_pred)
#         total_auc = roc_auc_score(Labels, Logits, average='micro')
#         total_auc_macro = roc_auc_score(Labels, Logits, average='macro')
#         print("Train Accuracy = {:.3f}".format(total_acc))
#         print("Train AUC = {:.3f}".format(total_auc))
#         print("Train AUC Macro = {:.3f}".format(total_auc_macro))