import torch
from utils.BEATs import BEATs, BEATsConfig

def load_beats_model(model_path: str):
    # load the pre-trained config
    checkpoint = torch.load(model_path)
    cfg = BEATsConfig(checkpoint['cfg'])
    # Set up the model
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()

    return BEATs_model