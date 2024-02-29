import copy
import os.path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torcheval.metrics import BinaryAUROC
from torcheval.metrics.aggregation.auc import AUC
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from tqdm import tqdm



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


def model_forward(model, config_dict, loss_fn, x, y):
    # x = torch.tensor(pd_x.iloc[data_idx, :].to_numpy()).to(current_device)
    # y = torch.tensor(pd_y.iloc[data_idx, :].to_numpy()).to(current_device)
    # x.requires_grad = True
    # y.requires_grad = True
    x = x.to(torch.float32)

    output = model(x).to(torch.float32)

    y = y.squeeze().to(torch.long)
    # loss = loss_fn(output.squeeze(), y.squeeze())
    y = torch.nn.functional.one_hot(y, num_classes=config_dict['num_classes']).to(torch.float32)
    loss = loss_fn(output, y)

    return output, loss

def train(model, config_dict, pd_x, pd_y, val_x=None, val_y=None):
    print("STATUS - Begining model training")
    n_epochs = config_dict['n_epochs']
    lr = config_dict['lr']
    batch_sz = config_dict['batch_size']
    optimizer = config_dict['optimizer']
    current_device = config_dict['device']
    loss_fn = config_dict['loss_fn']
    save_model_path = config_dict['save_model_path']
    model_name = config_dict['model_name']

    best_model = model

    if issubclass(model.__class__, BaseEstimator):
        model.fit(pd_x, pd_y)
    else:
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if current_device is None:
            current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"current_device: {current_device}")

        train_loader = format_pds(pd_x, pd_y, batch_sz, current_device)

        if not (val_x is None and val_y is None):
            val_loader = format_pds(val_x, val_y, batch_sz, current_device)

        model.train()
        model.to(torch.float32)
        model.to(current_device)

        best_aucroc = 0
        best_model_path = ""

        for epoch in range(1,n_epochs+1):
            print("---------------------------- START EPOCH ------------------------------")
            epoch_loss = 0
            e_iters = 0
            # for data_idx in range(0, len(pd_x)):
            with tqdm(train_loader, unit="batch", total=len(train_loader)) as tepoch:
                for x, y in tepoch:
                    tepoch.set_description(f"Epoch {epoch} with epoch_loss: {epoch_loss}")
                    e_iters += 1

                    #run the model
                    outs, loss = model_forward(model, config_dict, loss_fn, x, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

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

                val_outs = []
                val_true = []

                for x, y in tqdm(val_loader, unit="batch", total=len(val_loader)):
                    e_counts += 1

                    outs, loss = model_forward(model, config_dict, loss_fn, x, y)

                    val_outs.append(outs.detach().cpu().numpy())
                    val_true.append(y.detach().cpu().numpy())


                    preds = torch.argmax(outs, dim=-1).detach()
                    preds = preds.cpu().detach()
                    y = y.cpu().detach().squeeze()

                    incor = torch.sum(torch.abs(preds-y)).detach().item()
                    epoch_total += preds.size(0)
                    epoch_incorrect += incor
                    epoch_correct += preds.size(0) - incor

                    false_pos += torch.sum(np.logical_and(preds == 1, y == 0)).item()
                    false_neg += torch.sum(np.logical_and(preds == 0, y == 1)).item()
                    true_pos += torch.sum(np.logical_and(preds == 1, y == 1)).item()
                    true_neg += torch.sum(np.logical_and(preds == 0, y == 0)).item()

                    metric_auc.update(preds, y)
                    metric_aucroc.update(preds, y)

                    epoch_validation_loss += loss.item()

                #auc_val = metric_auc.compute()
                #aucroc_val = metric_aucroc.compute()
                from sklearn import metrics
                auc_roc = metrics.roc_auc_score(np.concatenate(val_true), np.concatenate(val_outs)[:, 1])
                #auc_roc = metrics.auc(np.concatenate(val_true), np.concatenate(val_outs)[:, 1])

                print(f"\nEpoch {epoch} \t\t Validation Loss: {epoch_validation_loss/e_counts} \t\t Total: {epoch_total} \t\t Correct: {epoch_correct} \t\t Fails: {epoch_incorrect}")
                print(f"Epoch {epoch} \t\t Accuracy: {epoch_correct/epoch_total}\t\t AUCROC: {auc_roc}")
                print(f"Epoch {epoch} \t\t false_pos: {false_pos}\t\t false_neg: {false_neg}\t\t true_pos: {true_pos}\t\t true_neg: {true_neg}\t\t")
                print("---------------------------- END EPOCH ------------------------------")
                metric_auc.reset()
                if auc_roc > best_aucroc:
                    best_aucroc = auc_roc

                    if os.path.exists(best_model_path):
                        os.remove(best_model_path)
                    new_save_file_path = save_model_path + f"{model_name}_epoch{epoch}_aucroc{auc_roc:.5f}.pt"
                    torch.save(model, new_save_file_path)
                    best_model_path = new_save_file_path

                    best_model = copy.deepcopy(model)
        print(f"FINAL\t\t Best AUCROC: {best_aucroc} stored in model at location {best_model_path}")
    return best_model