#!/bin/bash

# Base directory containing your audio folders
BASE_DIR="/zhome/58/f/181392/DTU/DL/Project/DL_RELAX/results"

# List of full paths of the audio files
audio_files=(
    "${BASE_DIR}/Regular_Temporal_Masking_137/1-137-A-32.wav"
    "${BASE_DIR}/Regular_Temporal_Masking_96890/1-96890-A-37.wav"
    "${BASE_DIR}/Regular_Frequency_Masking/1-22804-A-46.wav"
    "${BASE_DIR}/Irregular_Temporal_Frequency_Masking/1-34119-B-1.wav"
    "${BASE_DIR}/Irregular_Temporal_Masking/1-51805-C-33.wav"
    "${BASE_DIR}/Very_Hard_To_Guess/3-95695-A-5.wav"
    "${BASE_DIR}/Unclear_Classification/3-118972-B-41.wav"
    "${BASE_DIR}/Easy_Frequency/5-187201-B-4.wav"
)

# Loop through each audio file
for full_path in "${audio_files[@]}"; do
    # Submit a job to the HPC queue using the wrapper script
    /zhome/58/f/181392/DTU/DL/Project/DL_RELAX/submit_job.sh "$full_path"
done
