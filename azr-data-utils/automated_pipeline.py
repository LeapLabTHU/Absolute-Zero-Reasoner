#!/usr/bin/env python3
"""
AZR Automated Training Pipeline
Integrates data merging and completion detection into the AZR training workflow.

This script modifies the main AZR training script to:
1. Automatically merge existing data before training starts
2. Monitor training progress and stop when complete
3. Merge new data after training completes
"""

import sys
import os
import subprocess
import logging
import argparse
from pathlib import Path
import threading
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AZRPipeline:
    def __init__(self, 
                 azr_project_dir: str,
                 checkpoint_dir: str = None,
                 data_dir: str = "data/code_reason/"):
        self.azr_project_dir = Path(azr_project_dir)
        self.data_dir = Path(data_dir)
        
        if checkpoint_dir is None:
            self.checkpoint_dir = self.azr_project_dir / "checkpoints/code_io/azr/0.5b_coder_seed_generation/test_answer/Qwen2.5-Coder-0.5B/answer_conditional/code/"
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        
        self.training_script = self.azr_project_dir / "main_azr_ppo.py"
        self.utils_dir = Path(__file__).parent
        
    def run_data_merger(self, backup: bool = True) -> bool:
        """Run the parquet data merger."""
        merger_script = self.utils_dir / "merge_parquet_data.py"
        
        if not merger_script.exists():
            logger.error(f"Merger script not found: {merger_script}")
            return False
        
        cmd = [
            sys.executable, str(merger_script),
            "--checkpoint-dir", str(self.checkpoint_dir),
            "--output-dir", str(self.data_dir)
        ]
        
        if backup:
            cmd.append("--backup")
        
        logger.info(f"Running data merger: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, cwd=str(self.azr_project_dir), capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Data merger completed successfully")
                if result.stdout:
                    logger.info(result.stdout)
                return True
            else:
                logger.error(f"Data merger failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Failed to run data merger: {e}")
            return False
    
    def start_training_monitor(self, training_process) -> threading.Thread:
        """Start the training monitor in a separate thread."""
        monitor_script = self.utils_dir / "training_monitor.py"
        
        if not monitor_script.exists():
            logger.error(f"Monitor script not found: {monitor_script}")
            return None
        
        def run_monitor():
            cmd = [
                sys.executable, str(monitor_script),
                "--checkpoint-dir", str(self.checkpoint_dir),
                "--output-dir", str(self.data_dir),
                "--training-pid", str(training_process.pid),
                "--check-interval", "300",  # 5 minutes
                "--stagnation-threshold", "1800"  # 30 minutes
            ]
            
            logger.info("Starting training monitor...")
            try:
                subprocess.run(cmd, cwd=str(self.azr_project_dir))
            except Exception as e:
                logger.error(f"Training monitor error: {e}")
        
        monitor_thread = threading.Thread(target=run_monitor, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def run_training(self, training_args: list = None) -> bool:
        """Run the AZR training with monitoring."""
        if not self.training_script.exists():
            logger.error(f"Training script not found: {self.training_script}")
            return False
        
        # Prepare training command
        cmd = [sys.executable, str(self.training_script)]
        if training_args:
            cmd.extend(training_args)
        
        logger.info(f"Starting AZR training: {' '.join(cmd)}")
        
        try:
            # Start training process
            training_process = subprocess.Popen(
                cmd, 
                cwd=str(self.azr_project_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Start monitor
            monitor_thread = self.start_training_monitor(training_process)
            
            # Stream training output
            logger.info("Training started, streaming output...")
            for line in training_process.stdout:
                print(line.rstrip())
            
            # Wait for process to complete
            return_code = training_process.wait()
            
            if return_code == 0:
                logger.info("Training completed successfully")
                return True
            else:
                logger.error(f"Training failed with return code {return_code}")
                return False
                
        except Exception as e:
            logger.error(f"Training execution error: {e}")
            return False
    
    def check_prerequisites(self) -> bool:
        """Check if all required files and directories exist."""
        checks = [
            (self.azr_project_dir, "AZR project directory"),
            (self.training_script, "Training script"),
            (self.data_dir, "Data directory"),
        ]
        
        all_good = True
        for path, description in checks:
            if not path.exists():
                logger.error(f"{description} not found: {path}")
                all_good = False
            else:
                logger.info(f"✓ {description}: {path}")
        
        return all_good
    
    def run_full_pipeline(self, 
                         training_args: list = None,
                         pre_merge: bool = True,
                         post_merge: bool = True) -> bool:
        """Run the complete AZR training pipeline."""
        logger.info("=== AZR Automated Training Pipeline ===")
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed")
            return False
        
        # Pre-training data merge
        if pre_merge:
            logger.info("\n=== Phase 1: Pre-training Data Merge ===")
            if not self.run_data_merger(backup=True):
                logger.warning("Pre-training merge failed, continuing anyway...")
        
        # Run training with monitoring
        logger.info("\n=== Phase 2: Training with Auto-completion ===")
        success = self.run_training(training_args)
        
        # Post-training merge (if monitor didn't already do it)
        if post_merge and success:
            logger.info("\n=== Phase 3: Post-training Data Merge ===")
            if not self.run_data_merger(backup=False):
                logger.warning("Post-training merge failed")
        
        if success:
            logger.info("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
            logger.info("Training data has been consolidated and is ready for next iteration!")
        else:
            logger.error("\n=== PIPELINE FAILED ===")
        
        return success

def main():
    parser = argparse.ArgumentParser(description="Run automated AZR training pipeline")
    parser.add_argument("--azr-dir", required=True, help="AZR project directory")
    parser.add_argument("--checkpoint-dir", help="Custom checkpoint directory")
    parser.add_argument("--data-dir", default="data/code_reason/", help="Training data directory")
    parser.add_argument("--no-pre-merge", action="store_true", help="Skip pre-training data merge")
    parser.add_argument("--no-post-merge", action="store_true", help="Skip post-training data merge")
    parser.add_argument("--training-args", nargs="*", help="Additional arguments to pass to training script")
    
    args = parser.parse_args()
    
    pipeline = AZRPipeline(
        azr_project_dir=args.azr_dir,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir
    )
    
    success = pipeline.run_full_pipeline(
        training_args=args.training_args,
        pre_merge=not args.no_pre_merge,
        post_merge=not args.no_post_merge
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
