import torch
import torch.nn as nn


class V1Classifier(torch.nn.Module):
    def __init__(self, n_feats, n_classes):
        super().__init__()
        # self.l1 = nn.Linear(n_feats, n_feats*2)
        # self.l2 = nn.Linear(n_feats*2, n_feats * 3)
        # self.fc = nn.Linear(n_feats * 3, n_classes)
        # self.relu = nn.ReLU(inplace=False)
        # self.flattener = nn.Flatten()
        self.fc1 = nn.Linear(n_feats, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, n_classes)

        self.n_classes = n_classes

    def forward(self, in_x):
        # x = self.flattener(in_x)
        # x = self.l1(x)
        # x = self.relu(x)
        # x = self.l2(x)
        # x = self.relu(x)
        # x = self.fc(x)
        # x = self.relu(x)
        #max_idx = torch.argmax(x)
        #return torch.sigmoid(x)

        x = torch.flatten(in_x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x