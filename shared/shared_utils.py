import argparse
import pickle

import yaml
from yaml import CLoader
import random
import numpy as np
import torch

def set_random_seed(seed_val):
    print(f"Init - Setting random seed to {seed_val}")
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)


def pickel_results(obj, fname):
    pfile = open(fname, "wb")
    pickle.dump(obj, pfile)
    pfile.close()


def initialize_configuration():
    parser = argparse.ArgumentParser(
        prog='Explanation Testing Framework',
        description='Run a selection of explanation methods')
    parser.add_argument('configuration')
    args = parser.parse_args()
    config_path = args.configuration

    stream = open(config_path, "r")
    try:
        config = yaml.load(stream,  Loader=CLoader)
    finally:
        stream.close()

    config['num_classes'] = 2
    if not config['excludes'] is None:
        config['num_features'] = 18 - len(config['excludes'])
    else:
        config['num_features'] = 18
    config['n_classes'] = 2

    if config['training']['train']['loss_type'] == "NLL":
        config['loss_fn'] = torch.nn.NLLLoss()
    elif config['training']['train']['loss_type'] == "BCE":
        config['loss_fn'] = torch.nn.BCELoss()
    else:
        raise NotImplementedError(f"Loss type {config['loss_type']} is not implemented!")

    return config