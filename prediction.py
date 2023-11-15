import torch
from utils.BEATs import BEATs, BEATsConfig
import librosa
from utils import MaskGenerator

def load_audio(audio_path):
    # Load the audio file
    audio, sample_rate = librosa.load(audio_path)
    
    # Resample the audio file to 16kHz
    if sample_rate != 16000:
        return librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    return torch.from_numpy(audio).unsqueeze(0)

def extract_features(audio_path: str, model_path: str):
    checkpoint = torch.load(model_path)
    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()
    
    # load the pre-trained checkpoints
    checkpoint = torch.load(model_path)
    padding_mask = torch.zeros(1, 10000).bool()

    
    # load resampled audio 
    audio_input_16khz = load_audio(audio_path)
    representation = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]

    return representation

def saliency_var(x, model, saliency, num_batches, pdist, dev="cpu"):
    mask_bs = x.shape[0]
    input_size = x.shape[0]
    h_star = model(x)
    saliency_var = torch.zeros((input_size, input_size), device=dev)
    for _, mask in enumerate(MaskGenerator(num_batches, (input_size), mask_bs=mask_bs),
                            total=num_batches, desc=f"Compute model uncertainty"):

          x_mask = x * mask
          out = model(x_mask)

          out = pdist(h_star, out)[:, None, None, None]

          var = (out-saliency[None, None])**2
          var = torch.mean(var * mask, dim=(0, 1))

          saliency_var += var

    saliency_var /= ((num_batches-1)*0.5)

def saliency(x, model, num_batches, pdist, dev="cpu"):
    mask_bs = x.shape[0]
    input_size = x.shape[0]
    h_star = model(x)
    h_star = h_star.expand(mask_bs, -1)
    saliency = torch.zeros((input_size, input_size), device=dev)
    for _, mask in enumerate(MaskGenerator(
                                                num_batches, (input_size), mask_bs=mask_bs),
                                                total=num_batches,
                                                desc=f"Compute model importance"):

          x_mask = x * mask
          out = model(x_mask)

          out = pdist(h_star, out)[:, None, None, None]

          saliency += torch.mean(out * mask, dim=(0, 1))
   
    return saliency / (num_batches*0.5)


if __name__ == '__main__':
    filename = './audio/1-9886-A-49.wav'
    # input = load_audio(filename)

    model_path = './beats_env/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
    features = extract_features(audio_path=filename, model_path=model_path)
    print(features.shape)
    # num_batches = 80
    # pdist = torch.nn.CosineSimilarity(dim=1)
    # saliency(input, None, num_batches, pdist)
