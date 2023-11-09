import torch
from utils.BEATs import BEATs, BEATsConfig
import librosa

def extract_features(audio_path: str, model_path: str):
    checkpoint = torch.load(model_path)
    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()
    
    # load the pre-trained checkpoints
    checkpoint = torch.load(model_path)
    
    # Load the audio file
    audio, sample_rate = librosa.load(audio_path)
    padding_mask = torch.zeros(1, 10000).bool()
    
    # Resample the audio file to 16kHz
    if sample_rate != 16000:
        audio_input_16khz = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    audio_input_16khz = torch.from_numpy(audio).unsqueeze(0)
    representation = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]

    return representation

if __name__ == '__main__':
    filename = '/zhome/58/f/181392/DTU/DL/Project/DL_RELAX/audio/sounds/1-9886-A-49.wav'
    model_path = '/zhome/58/f/181392/DTU/DL/Project/DL_RELAX/audio/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
    features = extract_features(audio_path=filename, model_path=model_path)