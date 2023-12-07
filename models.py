import torch
import torch.nn as nn


def basic_model():
    # for imbalanced dataset:
    weights = [1 / (1 - (sum(train_y) / len(train_y))), 1 / (sum(train_y) / len(train_y))]
    class_weights = torch.FloatTensor(weights).cuda()
    balancedCE = nn.CrossEntropyLoss(weight=class_weights)
    # model params
    lr = 0.001
    model = V1Classifier(n_feats=(num_features - len(excludes)) * cutoff_seq_len, n_classes=2)
    basic_config = {
        'n_epochs': 10,
        'lr': lr,
        'batch_size': 32,
        'optimizer': torch.optim.Adam(model.parameters(), lr=lr),
        'device': current_device,
        'loss_fn': balancedCE,
    }
    #model = train(model, basic_config, train_x, train_y)

def gru_ode_bayes_model():
    gob_model_config = {
        'input_size': num_features,
        'hidden_size': 50,
        'p_hidden': 25,
        'prep_hidden': 10,
        'logvar': True,
        'mixing': 1e-4,
        'delta_t': 0.1,
        'T': 200,
        'lambda': 0,
        'classification_hidden': 2,
        'cov_size': cutoff_seq_len,
        'cov_hidden': 50,
        'dropout_rate': 0.2,
        'full_gru_ode': True,
        'no_cov': True,
        'impute': False,
        'lr': 0.001,
        'weight_decay': 0.0001,
    }
    model = gru_ode_bayes.NNFOwithBayesianJumps(input_size=gob_model_config["input_size"],
                                                hidden_size=gob_model_config["hidden_size"],
                                                p_hidden=gob_model_config["p_hidden"],
                                                prep_hidden=gob_model_config["prep_hidden"],
                                                logvar=gob_model_config["logvar"],
                                                mixing=gob_model_config["mixing"],
                                                classification_hidden=gob_model_config["classification_hidden"],
                                                cov_size=gob_model_config["cov_size"],
                                                cov_hidden=gob_model_config["cov_hidden"],
                                                dropout_rate=gob_model_config["dropout_rate"],
                                                full_gru_ode=gob_model_config["full_gru_ode"],
                                                impute=gob_model_config["impute"])
    lr = 0.001
    weight_decay = 0.0001
    gob_train_config = {
        'n_epochs': 10,
        'lr': lr,
        'batch_size': 32,
        'optimizer': torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
        'device': current_device,
        'loss_fn': torch.nn.BCEWithLogitsLoss(reduction='sum'),
    }
    model.to(gob_train_config['device'])
    #model = bayes_train(model, gob_train_config, train_x, train_y)

def tlstm_model():
    lr = 0.001
    basic_config = {
        'n_epochs': 10,
        'lr': lr,
        'batch_size': 32,
        'device': current_device,
        'loss_fn': balancedCE,
    }

    model = TLSTM(input_dim=15, output_dim=2, hidden_dim=512, fc_dim=64, key=1)
    cross_entropy, y_pred, y, logits, labels = model.get_cost_acc()

    basic_config['optimizer'] = tensorflow.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)
    #model = tf_train(model, basic_config, train_x, train_y)

def torchmimic_model():
    from torchmimic.benchmarks import IHMBenchmark

    root_dir = "data/in-hospital-mortality"

    from torchmimic.models import StandardLSTM

    model = StandardLSTM(
        n_classes=2,
        hidden_dim=16,
        num_layers=2,
        dropout_rate=0.3,
        bidirectional=False,
    )

    trainer = IHMBenchmark(
        model=model,
        train_batch_size=8,
        test_batch_size=256,
        data=root_dir,
        learning_rate=0.001,
        weight_decay=0,
        report_freq=200,
        device=0,
        sample_size=None,
        wandb=False,
    )

    #trainer.fit(100)


def benchmark_lstm_model(current_device, train_x, train_y, val_x, val_y):
    # model params
    lr = 0.001  # from paper
    dropout = 0.3  # their code says is best

    model = BasicLSTM(n_feats=15, n_classes=2)

    basic_config = {
        'n_epochs': 100,  # default from their code
        'lr': lr,
        'batch_size': 32,  # their code says is best
        'optimizer': torch.optim.Adam(model.parameters(), lr=lr),  # from paper
        'device': current_device,
        'loss_fn': nn.BCELoss(),  # default from their code
    }
    #model = train(model, basic_config, train_x, train_y, val_x, val_y)

def select_model(model_type):
    ### Basic Test Model #############################################################################################
    if model_type == "basic":
        basic_model()
    ### GRU ODE Bayes Model ##########################################################################################
    elif model_type == "gru_ode_bayes":
        gru_ode_bayes_model()
    ### T-LSTM Model ##########################################################################################
    elif model_type == "tlstm":
        tlstm_model()
    elif model_type == "torchmimic_lstm":
        torchmimic_model()
    ### Benchmark LSTM Model ##########################################################################################
    elif model_type == "benchmark_lstm":
        benchmark_lstm_model(current_device, train_x, train_y, val_x, val_y)
    elif model_type == "xgboost":
        import xgboost as xgb

        dtrain_reg = xgb.DMatrix(train_x, train_y, enable_categorical=True)
        dtest_reg = xgb.DMatrix(val_x, val_y, enable_categorical=True)

        params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}
        n = 100

        evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

        model = xgb.train(
            params=params,
            dtrain=dtrain_reg,
            num_boost_round=n,
            evals=evals,
        )
        from sklearn.metrics import mean_squared_error

        preds = model.predict(dtest_reg)
        rmse = mean_squared_error(y_test, preds, squared=False)
        print(f"RMSE of the base model: {rmse:.3f}")

    elif model_type == "brits":
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from pypots.data import load_specific_dataset
        from pypots.imputation import BRITS
        from pypots.utils.metrics import cal_mae

        # Data preprocessing. Tedious, but PyPOTS can help.
        # data = load_specific_dataset('physionet_2012')  # PyPOTS will automatically download and extract it.
        # X = data['X']
        # num_samples = len(X['RecordID'].unique())
        # X = X.drop(['RecordID', 'Time'], axis=1)
        # X = StandardScaler().fit_transform(X.to_numpy())
        # X = X.reshape(num_samples, 48, -1)
        # X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1)  # hold out 10% observed values as ground truth
        # X = masked_fill(X, 1 - missing_mask, np.nan)
        # dataset = {"X": X}
        # print(dataset["X"].shape)  # (11988, 48, 37), 11988 samples, 48 time steps, 37 features
        train_dataset = {"X": train_x, "Y": train_y}
        val_dataset = {"X": val_x, "Y": val_y}

        if os.path.isfile("brits_model_epochs_10"):
            filehandler = open("brits_model_epochs_10", 'rb')
            model = pickle.load(filehandler)
            filehandler.close()
        else:
            model = BRITS(n_steps=70, n_features=15, epochs=10, rnn_hidden_size=64)

            model.fit(train_dataset)
            # model.predict(test_set=val_dataset)
            wm = False
            mt = 'brits'

            filehandler = open("brits_model_epochs_10", 'wb')
            pickle.dump(model, filehandler)
            filehandler.close()

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