#!/usr/bin/env python
# scripts/train.py
"""
Standalone training script that doesn't require package installation.
Usage: python scripts/train.py --config config/test.yaml
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import argparse
from flow_factory.hparams.args import Arguments
from flow_factory.models.loader import load_model
from flow_factory.trainers.loader import load_trainer


def main():
    parser = argparse.ArgumentParser(description="Flow-Factory Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"[INFO] Loading config: {args.config}")
    config = Arguments.load_from_yaml(args.config)
    
    # Load model
    print(f"[INFO] Loading model: {config.model_args.model_type}")
    adapter = load_model(
        model_args=config.model_args,
        training_args=config.training_args,
    )
    
    # Load trainer
    print(f"[INFO] Loading trainer: GRPO")
    trainer = load_trainer(
        trainer_type="grpo",
        data_args=config.data_args,
        training_args=config.training_args,
        reward_args=config.reward_args,
        adapter=adapter,
    )
    
    # Resume if needed
    if args.resume:
        print(f"[INFO] Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("[INFO] Starting training...")
    trainer.run()


if __name__ == "__main__":
    main()