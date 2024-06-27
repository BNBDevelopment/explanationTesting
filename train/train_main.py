import torch
import numpy as np

from shared.shared_utils import set_random_seed, initialize_configuration
from train.train_data import load_file_data
from train.train_models import select_model
from train.train_train import train



def create_trained_model(config, data):
    if config['experiment']['load_model']:
        model = torch.load(config['experiment']['load_model_path'], map_location=torch.device(config['device']))
    else:
        model = select_model(config)
        config['optimizer'] = torch.optim.Adam(model.parameters(), lr=config['training']['train']['lr'])
        model = train(model, config, data['train_x'], data['train_y'], data['val_x'], data['val_y'])
    return model


def get_confidence_intervals_for_preds(config, model, test_x):
    get_avg_confidence = True
    if get_avg_confidence:
        from operator import add, truediv
        pos_logits = []
        neg_logits = []
        p_count = 0
        n_count = 0
        for x in test_x:
            logits = model(torch.from_numpy(x).to(config['device'], torch.float32).unsqueeze(0)).tolist()[0]
            if logits[0] > logits[1]:
                neg_logits.append(logits)
                n_count += 1
            else:
                pos_logits.append(logits)
                p_count += 1
        pos_ls = [x[0] for x in neg_logits]
        pos_arr = np.array(pos_ls)
        print(f"Median: {np.median(pos_arr)}")
        print(f"25, 50, 75 Percentiles: {np.percentile(pos_arr, [25, 50, 75])} ")


def main():
    print("STATUS - Initializing Configuration")
    config = initialize_configuration()
    set_random_seed(12345)

    print("STATUS - Starting Initial Data Load")
    train_x, train_y, feature_names = load_file_data(config, data_type="train")
    val_x, val_y, _ = load_file_data(config, data_type="val")
    test_x, test_y, _ = load_file_data(config, data_type="test")
    data = {
        "train_x":train_x,
        "train_y":train_y,
        "val_x":val_x,
        "val_y":val_y,
        "test_x":test_x,
        "test_y":test_y,
    }

    model = create_trained_model(config, data)

    #Get average confidence
    get_confidence_intervals_for_preds(config, model, test_x)

if __name__ == "__main__":
    main()