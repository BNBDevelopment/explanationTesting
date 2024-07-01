import numpy as np
import pandas as pd
import torch


class ModelWrapper():
    def __init__(self, model, n_classes=None, model_flag=None, batch_size=256, verbose=False, skip_autobatch=False, modeltype='pt',device='cpu'):
        super().__init__()
        self.model = model
        self.column_names = ["x1"]
        if n_classes is None:
            self.classes_ = np.array([x for x in range(model.n_classes)])
        else:
            self.classes_ = n_classes
        self.model_flag = model_flag
        self.batch_size = batch_size
        self.verbose = verbose
        self.skip_autobatch = skip_autobatch
        self.model_type = modeltype
        self.device = device

    def predict_label(self, x):
        if len(x.shape) == 2:
            x = x.reshape(-1, 48, 17)
        res = self.predict_proba(x)
        pred = np.argmax(res, axis=-1)
        return pred

    def predict(self, x):
        torch_x = torch.from_numpy(x).to(next(self.model.parameters()).device, torch.float32)
        pred = self(torch_x)
        return pred.cpu().detach().numpy()

    def __call__(self, x):
        #temp = self.predict(args[0])
        # return torch.from_numpy(temp).to(next(self.model.parameters()).device, torch.float32)
        return self.model(x)

    def predict_proba(self, x):
        if issubclass(x.__class__, torch.Tensor):
            x_input = x.to(self.model.fc.bias.device)
        elif issubclass(x.__class__, pd.DataFrame):
            x_input = x.to_numpy()
            x_input = torch.from_numpy(x_input).to(self.model.fc.bias.device).to(torch.float32)
        elif issubclass(x.__class__, np.ndarray):
            x_input = torch.from_numpy(x).to(self.device).to(torch.float32)
        else:
            raise NotImplementedError(f"Input class type {x.__class__} not covered for wrapped model")

        if not self.skip_autobatch:
            if x_input.size(0) > 128:
                if self.verbose:
                    print("NOTICE: Batching for explanation methods activated...")
                x_arr = torch.split(x_input, x_input.size(0)//(x_input.size(0)//self.batch_size), dim=0)
                reses = []
                for x_batch in x_arr:
                    res = self(x_batch)
                    reses.append(res.detach().cpu().numpy())
                res = np.concatenate(reses, axis=0)
            else:
                res = self(x_input)
                res = res.detach().cpu().numpy()
        else:
            res = self(x_input)
            res = res.detach().cpu().numpy()
        return res