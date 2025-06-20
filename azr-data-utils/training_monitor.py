#!/usr/bin/env python3
"""
AZR Training Completion Monitor
Monitors AZR training progress and automatically stops when no new data is being generated.

This script can be used to:
1. Monitor checkpoint directory for new data generation
2. Automatically stop training when data generation stagnates
3. Trigger data merging when training completes
"""

import time
import os
import psutil
import logging
import argparse
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingMonitor:
    def __init__(self, 
                 checkpoint_dir: str,
                 training_pid: Optional[int] = None,
                 check_interval: int = 300,  # 5 minutes
                 stagnation_threshold: int = 1800,  # 30 minutes
                 min_new_samples: int = 10):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.training_pid = training_pid
        self.check_interval = check_interval
        self.stagnation_threshold = stagnation_threshold
        self.min_new_samples = min_new_samples
        
        self.file_patterns = [
            'train_gen_code_f.parquet',
            'train_gen_code_i.parquet', 
            'train_gen_code_o.parquet',
            'train_pred_code_f.parquet',
            'train_pred_code_i.parquet',
            'train_pred_code_o.parquet'
        ]
        
        self.last_sizes = {}
        self.last_activity = datetime.now()
        self.initial_scan_done = False
        
    def scan_files(self) -> Dict[str, int]:
        """Scan parquet files and return row counts."""
        current_sizes = {}
        
        for pattern in self.file_patterns:
            file_path = self.checkpoint_dir / pattern
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    current_sizes[pattern] = len(df)
                except Exception as e:
                    logger.warning(f"Could not read {pattern}: {e}")
                    current_sizes[pattern] = 0
            else:
                current_sizes[pattern] = 0
        
        return current_sizes
    
    def detect_new_data(self, current_sizes: Dict[str, int]) -> Dict[str, int]:
        """Detect newly generated data since last check."""
        new_data = {}
        
        for pattern, current_size in current_sizes.items():
            last_size = self.last_sizes.get(pattern, 0)
            new_samples = current_size - last_size
            new_data[pattern] = new_samples
        
        return new_data
    
    def is_training_process_running(self) -> bool:
        """Check if training process is still running."""
        if self.training_pid is None:
            # Look for python processes running AZR training
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if 'main_azr_ppo.py' in cmdline or 'azr' in cmdline.lower():
                            logger.info(f"Found training process: PID {proc.info['pid']}")
                            self.training_pid = proc.info['pid']
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return False
        else:
            try:
                proc = psutil.Process(self.training_pid)
                return proc.is_running()
            except psutil.NoSuchProcess:
                return False
    
    def stop_training_process(self) -> bool:
        """Gracefully stop the training process."""
        if self.training_pid is None:
            logger.warning("No training PID known, cannot stop process")
            return False
        
        try:
            proc = psutil.Process(self.training_pid)
            logger.info(f"Sending SIGTERM to training process {self.training_pid}")
            proc.terminate()
            
            # Wait for graceful shutdown
            try:
                proc.wait(timeout=30)
                logger.info("Training process stopped gracefully")
                return True
            except psutil.TimeoutExpired:
                logger.warning("Process did not stop gracefully, sending SIGKILL")
                proc.kill()
                return True
        except psutil.NoSuchProcess:
            logger.info("Training process already stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop training process: {e}")
            return False
    
    def run_data_merger(self, output_dir: str) -> bool:
        """Run the data merger script."""
        try:
            import subprocess
            
            merger_script = Path(__file__).parent / "merge_parquet_data.py"
            if not merger_script.exists():
                logger.error(f"Merger script not found: {merger_script}")
                return False
            
            cmd = [
                sys.executable, str(merger_script),
                "--checkpoint-dir", str(self.checkpoint_dir),
                "--output-dir", output_dir,
                "--backup"
            ]
            
            logger.info(f"Running data merger: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Data merger completed successfully")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"Data merger failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to run data merger: {e}")
            return False
    
    def monitor(self, auto_stop: bool = True, auto_merge: bool = True, output_dir: str = "data/code_reason/"):
        """Main monitoring loop."""
        logger.info("=== AZR Training Monitor Started ===")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Check interval: {self.check_interval}s")
        logger.info(f"Stagnation threshold: {self.stagnation_threshold}s")
        logger.info(f"Auto-stop: {auto_stop}, Auto-merge: {auto_merge}")
        
        consecutive_no_activity = 0
        
        try:
            while True:
                # Check if training process is still running
                training_running = self.is_training_process_running()
                
                if not training_running and not self.initial_scan_done:
                    logger.warning("No training process found, but continuing to monitor...")
                
                # Scan current file sizes
                current_sizes = self.scan_files()
                
                # Log current status
                total_samples = sum(current_sizes.values())
                logger.info(f"Current dataset size: {total_samples} total samples")
                for pattern, size in current_sizes.items():
                    if size > 0:
                        logger.info(f"  {pattern}: {size} samples")
                
                if self.initial_scan_done:
                    # Detect new data
                    new_data = self.detect_new_data(current_sizes)
                    total_new = sum(new_data.values())
                    
                    if total_new >= self.min_new_samples:
                        logger.info(f"Detected {total_new} new samples")
                        for pattern, count in new_data.items():
                            if count > 0:
                                logger.info(f"  {pattern}: +{count} new samples")
                        self.last_activity = datetime.now()
                        consecutive_no_activity = 0
                    else:
                        consecutive_no_activity += 1
                        time_since_activity = datetime.now() - self.last_activity
                        logger.info(f"No new data detected (stagnant for {time_since_activity})")
                        
                        # Check if we should stop training
                        if (time_since_activity.total_seconds() > self.stagnation_threshold and 
                            auto_stop and training_running):
                            logger.warning(f"Training stagnant for {time_since_activity}, stopping training...")
                            
                            if self.stop_training_process():
                                logger.info("Training stopped successfully")
                                
                                if auto_merge:
                                    logger.info("Starting automatic data merge...")
                                    if self.run_data_merger(output_dir):
                                        logger.info("=== TRAINING COMPLETION SEQUENCE FINISHED ===")
                                        logger.info("Training data has been merged and is ready for next run!")
                                    else:
                                        logger.error("Data merge failed, manual intervention required")
                                
                                break
                            else:
                                logger.error("Failed to stop training process")
                
                # Update file sizes for next iteration
                self.last_sizes = current_sizes.copy()
                self.initial_scan_done = True
                
                # Check if training process stopped naturally
                if training_running and not self.is_training_process_running():
                    logger.info("Training process stopped naturally")
                    if auto_merge:
                        logger.info("Starting automatic data merge...")
                        if self.run_data_merger(output_dir):
                            logger.info("=== TRAINING COMPLETION SEQUENCE FINISHED ===")
                        else:
                            logger.error("Data merge failed")
                    break
                
                # Wait before next check
                logger.info(f"Waiting {self.check_interval}s before next check...")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("Monitor stopped by user")
        except Exception as e:
            logger.error(f"Monitor error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Monitor AZR training and auto-stop when complete")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing checkpoint parquet files")
    parser.add_argument("--output-dir", default="data/code_reason/", help="Output directory for merged data")
    parser.add_argument("--training-pid", type=int, help="PID of training process to monitor")
    parser.add_argument("--check-interval", type=int, default=300, help="Check interval in seconds (default: 300)")
    parser.add_argument("--stagnation-threshold", type=int, default=1800, help="Stagnation threshold in seconds (default: 1800)")
    parser.add_argument("--min-new-samples", type=int, default=10, help="Minimum new samples to consider active (default: 10)")
    parser.add_argument("--no-auto-stop", action="store_true", help="Disable automatic training stop")
    parser.add_argument("--no-auto-merge", action="store_true", help="Disable automatic data merging")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_dir):
        logger.error(f"Checkpoint directory does not exist: {args.checkpoint_dir}")
        sys.exit(1)
    
    monitor = TrainingMonitor(
        checkpoint_dir=args.checkpoint_dir,
        training_pid=args.training_pid,
        check_interval=args.check_interval,
        stagnation_threshold=args.stagnation_threshold,
        min_new_samples=args.min_new_samples
    )
    
    monitor.monitor(
        auto_stop=not args.no_auto_stop,
        auto_merge=not args.no_auto_merge,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
