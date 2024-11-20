#!/usr/bin/env python3
"""
Main training script for MERT-based genre classification.
Provides CLI interface for training and evaluation.
"""

import os
import logging
import argparse
import wandb
from pathlib import Path
from transformers import TrainingArguments, Trainer
from transformers.trainer_callback import EarlyStoppingCallback

from config import MERT_CONFIGS, TRAINING_CONFIG, DATASET_CONFIG
from model_MERT import create_model, GTZANDataset
from utils import LossLoggingCallback, compute_metrics, plot_confusion_matrix

def setup_logging(log_file=None, debug=False):
    """Configure logging settings."""
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(level=log_level, format=log_format)
    logger = logging.getLogger()
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MERT-based genre classifier.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Root directory containing the dataset')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save results and checkpoints')
    
    # Model arguments
    parser.add_argument('--model-version', type=str, choices=['95m', '330m'],
                        default='95m', help='MERT model version to use')
    
    # Training arguments
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs (default: model-specific)')
    parser.add_argument('--batch-size', type=int,
                        help='Training batch size (default: model-specific)')
    parser.add_argument('--learning-rate', type=float,
                        help='Learning rate (default: model-specific)')
    parser.add_argument('--warmup-steps', type=int,
                        default=TRAINING_CONFIG['warmup_steps'],
                        help='Number of warmup steps')
    parser.add_argument('--weight-decay', type=float,
                        default=TRAINING_CONFIG['weight_decay'],
                        help='Weight decay')
    parser.add_argument('--early-stopping-patience', type=int,
                        default=TRAINING_CONFIG['early_stopping_patience'],
                        help='Patience for early stopping')
    
    # Logging arguments
    parser.add_argument('--wandb-project', type=str,
                        default='mert-genre-classifier',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-run-name', type=str,
                        help='Weights & Biases run name (default: auto-generated)')
    parser.add_argument('--log-file', type=str,
                        help='Path to log file (optional)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    setup_logging(args.log_file, args.debug)
    
    # Get model configuration
    model_config = MERT_CONFIGS[args.model_version]
    
    # Set default values based on model version if not provided
    epochs = args.epochs or model_config['default_epochs']
    batch_size = args.batch_size or model_config['default_batch_size']
    learning_rate = args.learning_rate or model_config['default_lr']
    
    # Initialize wandb
    run_name = args.wandb_run_name or f"mert-{args.model_version}-genre-classifier"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model_version": args.model_version,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "early_stopping_patience": args.early_stopping_patience
        }
    )

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize model and processor
        logging.info(f"Initializing MERT-{args.model_version} model and processor")
        processor, model = create_model(args.model_version)
        
        # Create datasets
        logging.info("Loading datasets...")
        train_dataset = GTZANDataset(args.data_dir, 'train', processor)
        val_dataset = GTZANDataset(args.data_dir, 'valid', processor)
        test_dataset = GTZANDataset(args.data_dir, 'test', processor)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_strategy="steps",
            save_steps=TRAINING_CONFIG['save_steps'],
            evaluation_strategy="steps",
            eval_steps=TRAINING_CONFIG['eval_steps'],
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=TRAINING_CONFIG['logging_steps'],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            report_to=["wandb"],
            save_total_limit=TRAINING_CONFIG['save_total_limit'],
        )
        
        # Initialize trainer
        loss_callback = LossLoggingCallback()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience,
                    early_stopping_threshold=TRAINING_CONFIG['early_stopping_threshold']
                ),
                loss_callback
            ],
        )
        
        # Training
        logging.info(f"Starting training with MERT-{args.model_version}...")
        trainer.train()
        
        # Evaluation
        logging.info("\nGenerating final evaluation results...")
        val_results = trainer.evaluate(val_dataset)
        test_results = trainer.evaluate(test_dataset)
        
        logging.info(f"Validation Results: {val_results}")
        logging.info(f"Test Results: {test_results}")
        
        # Generate confusion matrix
        test_predictions = trainer.predict(test_dataset)
        y_true = test_predictions.label_ids
        y_pred = test_predictions.predictions.argmax(-1)
        
        genre_labels = [train_dataset.reverse_mapping[i] for i in range(DATASET_CONFIG['num_classes'])]
        confusion_matrix_path = os.path.join(
            args.output_dir, 
            f"confusion_matrix_{args.model_version}.png"
        )
        
        confusion_fig = plot_confusion_matrix(
            y_true,
            y_pred,
            genre_labels,
            f"Genre Classification Confusion Matrix (MERT-{args.model_version})"
        )
        
        confusion_fig.savefig(
            confusion_matrix_path,
            dpi=300,
            bbox_inches='tight'
        )
        
        # Log final results to wandb
        wandb.log({
            "confusion_matrix": wandb.Image(confusion_matrix_path),
            "final_val_results": val_results,
            "final_test_results": test_results,
            "class_distribution": {
                genre_labels[i]: np.sum(y_true == i) 
                for i in range(DATASET_CONFIG['num_classes'])
            }
        })
        
        logging.info(f"Results saved to {args.output_dir}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
