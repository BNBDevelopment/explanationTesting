import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from tqdm import tqdm


def train(model, pd_x, pd_y, n_epochs=10, lr=0.01, batch_sz=32, optimizer=None, current_device=None, loss_fn=nn.functional.mse_loss):

    if issubclass(model.__class__, BaseEstimator):
        model.fit(pd_x, pd_y)
    else:
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if current_device is None:
            current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"current_device: {current_device}")


        #create torch data
        x = torch.tensor(pd_x.to_numpy()).to(current_device)
        y = torch.tensor(pd_y.to_numpy()).to(current_device)
        ds_set = torch.utils.data.TensorDataset(x, y)
        d_loader = torch.utils.data.DataLoader(ds_set, batch_size=batch_sz, shuffle=True, drop_last=True)


        model.train()
        model.to(torch.float32)
        model.to(current_device)

        for epoch in range(1,n_epochs+1):
            epoch_loss = 0
            e_iters = 0
            # for data_idx in range(0, len(pd_x)):
            for x, y in tqdm(d_loader, unit="batch"):
                e_iters += 1

                #x = torch.tensor(pd_x.iloc[data_idx, :].to_numpy()).to(current_device)
                #y = torch.tensor(pd_y.iloc[data_idx, :].to_numpy()).to(current_device)
                #x.requires_grad = True
                #y.requires_grad = True
                x = x.to(torch.float32)
                y = y.squeeze().to(torch.long)

                output = model(x).to(torch.float32)

                #loss = loss_fn(output.squeeze(), y.squeeze())
                loss = loss_fn(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()

            print(f"epoch loss: {epoch_loss/e_iters}")

    return model