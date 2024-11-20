# Low Resource Music Genre Classification

## Creating Dataset

You can use the following script to automatically create 30-second samples from your songs. You can adjust the number of samples that you want to create from one song.

1. Collect your songs. Make sure the format is `.wav`.
2. Use the `sample_audio.py` script to automatically create samples from one song. Use the following command:

   ```bash
   python sample_audio.py --input_path 'path/to/input' --output_path 'path/to/output' --num_samples 5 --sample_rate 16000


## Fine-tune AST
1. Download Dataset https://mbzuaiac-my.sharepoint.com/:f:/g/personal/mohammed_zumri_mbzuai_ac_ae/EgUL57_jk0VJtTOLwYzzpiUBHJh3SXb1LDYra6lhHvuMtw?e=sWzFHj

2. Install Dependencies
   ```bash
   pip3 install -r requirement.txt

3. Pull pre-trained models
   ```bash
   git lfs fetch --all

4. For training with IDS-NMR
   ```bash
   python3 main.py --n_epochs 
   --batch_size
   --data_path /path/to/dataset 
   --model_save_path /path/to/bestmodel
   --reprog_front skip 
   --lr 1e-4

   You can vary the learning rate, batch size as needed

## Fine-tune MERT
1. Go to MERT directory
2. do the following to start finetune
   ```bash
   python train.py 
    --data-dir /path/to/dataset 
    --output-dir ./results 
    --model-version 330m 
    --epochs 10 
    --batch-size 2 
    --learning-rate 1e-5 
    --warmup-steps 1000 
    --weight-decay 0.02 
    --early-stopping-patience 30 
    --wandb-project my-project 
    --wandb-run-name my-experiment 
    --log-file training.log 
    --debug
