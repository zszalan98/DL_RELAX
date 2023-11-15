import torch
from utils.BEATs import BEATs, BEATsConfig
import librosa

def load_beats_model(model_path: str):
    # load the pre-trained config
    checkpoint = torch.load(model_path)
    cfg = BEATsConfig(checkpoint['cfg'])
    # Set up the model
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()

    return BEATs_model

def resample_audio(audio_path):
    # Load the audio file
    audio, sample_rate = librosa.load(audio_path)
    
    # Resample the audio file to 16kHz
    if sample_rate != 16000:
        return librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    return torch.from_numpy(audio).unsqueeze(0)

def extract_features(audio_path: str, model_path: str):
    BEATs_model = load_beats_model(model_path=model_path)
    
    # Load and resample the audio file
    audio_input_16khz = resample_audio(audio_path=audio_path)
    
    padding_mask = torch.zeros(1, 10000).bool()
    
    lprobs, padding_mask, latent_representation = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)

    return lprobs, padding_mask, latent_representation

if __name__ == '__main__':
    filename = './audio/1-9886-A-49.wav'
    model_path = './beats_env/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
    _, _, features = extract_features(audio_path=filename, model_path=model_path)
