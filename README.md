# Low Resource Music Genre Classification
## Creating Dataset
You can use the following script to automatically create 30s sample from your songs. You can adjust the number of sample that you want to create from one song.
1. Collect your songs. Make sure the format is .wav
2. Use the `sample_audio.py` to automatically create samples from one song. This is the command `python sample_audio.py --input_path 'path/to/input' --output_path 'path/to/output' --num_samples 5 --sample_rate 16000`

## Fine-tune AST

## Fine-tune MERT
1. Go to MERT directory
2. do the following to start finetune
   python train.py \
    --data-dir /path/to/dataset \
    --output-dir ./results \
    --model-version 330m \
    --epochs 10 \
    --batch-size 2 \
    --learning-rate 1e-5 \
    --warmup-steps 1000 \
    --weight-decay 0.02 \
    --early-stopping-patience 30 \
    --wandb-project my-project \
    --wandb-run-name my-experiment \
    --log-file training.log \
    --debug
