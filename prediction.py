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

def saliency(x, model, mask_bs, inp_shape, num_batches, pdist, dev="cpu"):
    h_star = model(x)
    h_star = h_star.expand(mask_bs, -1)
    saliency = torch.zeros((inp_shape, inp_shape), device=dev)
    saliency_var = torch.zeros((inp_shape, inp_shape), device=dev)
    for mask_idx, mask in enumerate(tqdm.tqdm(MaskGenerator(
                                                num_batches, (inp_shape), mask_bs=mask_bs),
                                                total=num_batches,
                                                desc=f"Compute {model_name} importance")):

          x_mask = x * mask
          out = model(x_mask)

          out = pdist(h_star, out)[:, None, None, None]

          saliency += torch.mean(out * mask, dim=(0, 1))
   
    return saliency / (num_batches*0.5)

if __name__ == '__main__':
    filename = '/zhome/58/f/181392/DTU/DL/Project/DL_RELAX/audio/sounds/1-9886-A-49.wav'
    model_path = '/zhome/58/f/181392/DTU/DL/Project/DL_RELAX/audio/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
    features = extract_features(audio_path=filename, model_path=model_path)