# The path to the audio file
AUDIO_FILE=$1

module load cuda/11.6
# Load the Python module
source /zhome/58/f/181392/DTU/DL/Project/DL_RELAX/beats_env/bin/activate
# Run the Python script with the audio file
python /zhome/58/f/181392/DTU/DL/Project/DL_RELAX/get_results.py -f "$AUDIO_FILE"