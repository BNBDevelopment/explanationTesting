import torch
import torch.nn as nn


def basic_model(n_feats, n_classes):
    # for imbalanced dataset:
    weights = [1 / (1 - (sum(train_y) / len(train_y))), 1 / (sum(train_y) / len(train_y))]
    class_weights = torch.FloatTensor(weights).cuda()
    balancedCE = nn.CrossEntropyLoss(weight=class_weights)
    # model params
    lr = 0.001
    model = V1Classifier(n_feats=(n_feats - len(excludes)) * cutoff_seq_len, n_classes=2)
    basic_config = {
        'n_epochs': 10,
        'lr': lr,
        'batch_size': 32,
        'optimizer': torch.optim.Adam(model.parameters(), lr=lr),
        'device': current_device,
        'loss_fn': balancedCE,
    }
    #model = train(model, basic_config, train_x, train_y)

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        init_size = 17*48
        self.flattener = torch.nn.Flatten()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(init_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, init_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(self.flattener(x))
        decoded = self.decoder(encoded).reshape(-1, 48, 17)
        return decoded


def select_model(config):
    num_classes = config['num_classes']
    num_layers = config['training']['model']['model_n_layers']
    dropout = config['training']['model']['model_dropout']
    bias = config['training']['model']['model_bias']
    bidirectional = config['training']['model']['model_bidirectional']
    hidden_size = config['training']['model']['model_hdim']

    if config['training']['data']['data_preproc'] == 'mimic3benchmark':
        num_features = 76
    else:
        if config['training']['data']['input_concat_w_mask']:
            num_features = config['num_features'] * 2
        else:
            num_features = config['num_features']

    model_class = config['training']['model']["model_class"]
    if model_class == "basic":
        m = basic_model(n_feats=num_features, n_classes=num_classes)
        return m
    elif model_class == "benchmark_lstm":
        m = BasicLSTM(n_feats=num_features, n_classes=num_classes, hidden_size=hidden_size, num_layers=num_layers, bias=bias, dropout=dropout, bidirectional=bidirectional)
        m.lstm_layer.flatten_parameters()
        return m
    elif model_class == "channelwise_lstm":
        m =  ChannelWiseLSTM(n_feats=num_features, n_classes=num_classes, hidden_size=hidden_size, num_layers=num_layers, bias=bias, dropout=dropout, bidirectional=bidirectional)
        m.lstm_layer.flatten_parameters()
        return m

class V1Classifier(torch.nn.Module):
    def __init__(self, n_feats, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(n_feats, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, n_classes)

        self.n_classes = n_classes

    def forward(self, in_x):
        x = torch.flatten(in_x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class BasicLSTM(torch.nn.Module):
    def __init__(self, n_feats, n_classes, hidden_size=32, num_layers=2, bias=True, dropout=0.1, bidirectional=True):
        super().__init__()
        self.n_classes = n_classes
        self.lstm_layer = torch.nn.LSTM(input_size=n_feats, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=True, dropout=dropout,
                     bidirectional=bidirectional, proj_size=0)
        h_dim = hidden_size
        if bidirectional:
            h_dim = hidden_size * 2

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(h_dim, n_classes)

    def forward(self, in_x):
        z, hs = self.lstm_layer(in_x)
        final_h = z[:,-1,:]

        self.drop(final_h)
        final_h = self.fc(final_h)
        return torch.sigmoid(final_h)


class ChannelWiseLSTM(torch.nn.Module):
    def __init__(self, n_feats, n_classes, hidden_size=32, num_layers=1, bias=True, dropout=0.1, bidirectional=True):
        super().__init__()
        self.n_classes = n_classes

        self.lstms = nn.ModuleList()
        for i in range(int(n_feats)):
            lstm_layer = torch.nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=True, dropout=dropout,
                     bidirectional=bidirectional, proj_size=0)
            self.lstms.append(lstm_layer)

        self.merge_lstm = torch.nn.LSTM(input_size=n_feats, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=True, dropout=dropout,
                     bidirectional=bidirectional, proj_size=0)

        h_size = hidden_size
        if bidirectional:
            h_size = h_size * 2
        fc_in_size = h_size * int(n_feats)

        self.fc = nn.Linear(h_size, n_classes)

    def forward(self, in_x):
        per_channel_res = []
        for channel_idx, lstm_layer in enumerate(self.lstms):
            z, hs = lstm_layer(in_x[:,:,channel_idx].unsqueeze(-1))
            final_h = z[:,-1,:]
            per_channel_res.append(final_h)

        per_channel_full = torch.stack(per_channel_res, dim=-1)
        z, hs = self.merge_lstm(per_channel_full)
        final_h = z[:, -1, :]

        final_h = self.fc(final_h)
        return torch.sigmoid(final_h)