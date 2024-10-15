import os
import random
import argparse
from pydub import AudioSegment

# Function to sample a 30-second clip from an audio file
def sample_audio(audio_file, sample_duration=30*1000):
    audio = AudioSegment.from_wav(audio_file)
    audio_length = len(audio)
    
    # Ensure the audio file is at least 30 seconds long
    if audio_length < sample_duration:
        raise ValueError(f"Audio file {audio_file} is shorter than {sample_duration / 1000} seconds.")
    
    # Randomly select a start point for the 30-second clip
    start_point = random.randint(0, audio_length - sample_duration)
    return audio[start_point:start_point + sample_duration]

# Main function to process each file in the folder
def process_audio_folder(input_folder, output_folder, num_samples=3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            input_file = os.path.join(input_folder, file_name)
            
            # Process each file to generate the number of samples specified
            for i in range(num_samples):
                try:
                    clip = sample_audio(input_file)
                    
                    # Save the sampled clip to the output folder
                    output_file = os.path.join(output_folder, f"{file_name[:-4]}_sample_{i+1}.wav")
                    clip.export(output_file, format="wav")
                    print(f"Saved sample {i+1} for {file_name}")
                except ValueError as e:
                    print(e)

# Command-line interface using argparse
def main():
    parser = argparse.ArgumentParser(description="Sample 30-second clips from .wav files.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input folder containing .wav files.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output folder to save sampled clips.')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of 30-second samples to extract from each .wav file.')

    args = parser.parse_args()

    input_folder = args.input_path
    output_folder = args.output_path
    num_samples = args.num_samples

    process_audio_folder(input_folder, output_folder, num_samples)

if __name__ == "__main__":
    main()