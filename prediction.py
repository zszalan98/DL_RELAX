import torch
from utils.BEATs import BEATs, BEATsConfig

model_path = '/zhome/58/f/181392/DTU/DL/Project/DL_RELAX/audio/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
# load the pre-trained checkpoints
checkpoint = torch.load(model_path)

cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()

# extract the the audio representation
audio_input_16khz = torch.randn(1, 10000)
padding_mask = torch.zeros(1, 10000).bool()

representation = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]