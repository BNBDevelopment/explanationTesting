import torch.nn
import torch.nn as nn

class LSTMClassification(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, target_size, device, window_size, num_layers):
        super(LSTMClassification, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_layers).to(device)
        self.flattener = torch.nn.Flatten(start_dim=1, end_dim=2)
        self.fc = nn.Linear(hidden_dim * window_size, target_size).to(device)

    def forward(self, xinput, hs=None):
        lstm_out, (h, c) = self.lstm(xinput)

        out = self.flattener(lstm_out)
        #logits = self.fc(out)
        #scores = nn.functional.sigmoid(logits)
        #return scores

        prediction = self.fc(out)
        return prediction, out