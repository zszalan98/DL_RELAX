# Import
import torch
from tools.audio import prepare_audio
from tools.beats import load_beats_model

# Settings classes
class RelaxSettings:
    num_of_batches: int = 10  # Number of batches
    num_of_masks: int = 100  # Number of masks per batch
    
class AudioSettings:
    pass

class MaskingSettings:
    seed: int = 42  # Random seed (needed due to batched processing)


class AllSettings:
    def __init__(self):
        self.relax = RelaxSettings()
        self.audio = AudioSettings()
        self.masking = MaskingSettings()


# Run batched relax function
def run_batched_relax(home_path: Path, audio_filename: str, settings: AllSettings):
    ## Settings
    num_batches = settings.relax.num_of_batches
    num_masks = settings.relax.num_of_masks

    ## INITIALIZATION
    # I. Audio processing
    audio_folder = home_path.joinpath('dataset/selected')
    audio_path = audio_folder.joinpath(audio_filename)

    audio, cpx_spec, spec_db, spec_shape = prepare_audio(audio_path)

    # II. BEATS model
    beats_folder = home_path.joinpath('beats')
    beats_model_path = beats_folder.joinpath('BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')

    beats_model = load_beats_model(beats_model_path)

    # III. Init RELAX
    importance_mx = torch.zeros(spec_shape)
    similarity_mx = torch.zeros((num_batches, num_masks))
    _, _, h_star, _ = beats_model.extract_features(audio)

    ## RUN IN BATCHES
    # Compute RELAX - Step 1: Importance
    for b in range(settings.relax.num_of_batches):
        # 1. Create masks
        masks = create_masks(settings.masking)
        # 2. Apply masks
        masked_audio_signals = apply_masks(cpx_spec, masks)
        # 3. Extract features
        _, _, h_masked, _ = beats_model.extract_features(masked_audio_signals)
        # 4. Compute similarity
        s = cosine_sim(h_star, h_masked)
        similarity_mx[b, :] = s
        # 5. Update RELAX (importance)
        importance_mx = update_relax(importance_mx, masks, s)

    # Compute RELAX - Step 2: Uncertainty
    

    


# Path handling
from pathlib import Path
home_path = Path(__file__).parent  # Get parent folder of this file

## MAIN PROGRAM
audio_filename = '1-9886-A-49.wav'