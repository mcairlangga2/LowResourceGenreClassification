# Low Resource Music Genre Classification
## Creating Dataset
You can use the following script to automatically create 30s sample from your songs. You can adjust the number of sample that you want to create from one song.
1. Collect your songs. Make sure the format is .wav
2. Use the `sample_audio.py` to automatically create samples from one song. This is the command `python sample_audio.py --input_path 'path/to/input' --output_path 'path/to/output' --num_samples 5 --sample_rate 16000`
