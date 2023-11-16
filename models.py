import torch
import torch.nn as nn


class V1Classifier(torch.nn.Module):
    def __init__(self, n_feats, n_classes):
        super().__init__()
        self.l1 = nn.Linear(n_feats, n_feats*2)
        self.l2 = nn.Linear(n_feats*2, n_feats * 3)
        self.l3 = nn.Linear(n_feats * 3, n_classes)
        self.relu = nn.ReLU(inplace=False)
        self.flattener = nn.Flatten()

    def forward(self, x):
        x = self.flattener(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)

        #max_idx = torch.argmax(x)
        return torch.sigmoid(x)