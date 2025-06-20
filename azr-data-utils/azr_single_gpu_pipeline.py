#!/usr/bin/env python3
"""
AZR Single-GPU Training Pipeline
Combines data consolidation with training monitoring for automated AZR training.

Features:
- Uses proven consolidation logic from original script
- Adds training monitoring and auto-stopping
- Maintains visual output and AZR-specific paths
- Handles both seed generation and self-play data
"""

import pandas as pd
import os
import shutil
import time
import psutil
import subprocess
import threading
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

class AZRSingleGPUPipeline:
    def __init__(self):
        # AZR project paths (hardcoded for simplicity)
        self.base_dir = "/home/frankshortt/AI/qwen3-training/Absolute-Zero-Reasoner"
        self.current_train_file = os.path.join(self.base_dir, "data/code_reason/test_answer.parquet")
        
        # Checkpoint directories
        self.seed_dir = os.path.join(self.base_dir, "checkpoints/code_io/azr/0.5b_coder_seed_generation/test_answer/Qwen2.5-Coder-0.5B/answer_conditional/code/")
        self.selfplay_dir = os.path.join(self.base_dir, "checkpoints/code_io/azr/0.5b_coder_selfplay_single_gpu/test_answer/Qwen2.5-Coder-0.5B/answer_conditional/code/")
        
        # Training script
        self.training_script = os.path.join(self.base_dir, "main_azr_ppo.py")
        
        # Monitoring settings
        self.check_interval = 300  # 5 minutes
        self.stagnation_threshold = 1800  # 30 minutes
        self.min_new_samples = 10
        
        self.monitoring = False
        self.training_process = None
        self.last_file_sizes = {}
        self.last_activity = datetime.now()

    def create_backup(self):
        """Create backup of current training file."""
        backup_file = os.path.join(
            self.base_dir, 
            f"data/code_reason/test_answer_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        )
        print(f"📦 Creating backup: {os.path.basename(backup_file)}")
        shutil.copy2(self.current_train_file, backup_file)
        return backup_file

    def consolidate_data(self):
        """Enhanced version of the original consolidation script."""
        print("🔍 AZR Data Consolidation Starting...")
        print("=" * 60)
        
        # Step 1: Backup current training file
        backup_file = self.create_backup()
        
        # Step 2: Load current training data
        print("📊 Loading current training data...")
        df_current = pd.read_parquet(self.current_train_file)
        print(f"   Current training data: {len(df_current):,} rows")
        
        # Step 3: Load and combine all generated data
        all_dataframes = [df_current]
        
        # Seed generation files
        print("\n🌱 Processing seed generation data...")
        seed_files = ["train_gen_code_f.parquet", "train_gen_code_i.parquet", "train_gen_code_o.parquet"]
        seed_total = 0
        
        for filename in seed_files:
            filepath = os.path.join(self.seed_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_parquet(filepath)
                print(f"   📁 {filename}: {len(df):,} rows")
                all_dataframes.append(df)
                seed_total += len(df)
            else:
                print(f"   ⚠️  {filename}: Not found")
        
        print(f"   🌱 Total seed generation: {seed_total:,} rows")
        
        # Self-play files
        print("\n🎮 Processing self-play data...")
        selfplay_files = [
            "train_gen_code_f.parquet", "train_gen_code_i.parquet", "train_gen_code_o.parquet",
            "train_pred_code_f.parquet", "train_pred_code_i.parquet", "train_pred_code_o.parquet"
        ]
        selfplay_total = 0
        
        for filename in selfplay_files:
            filepath = os.path.join(self.selfplay_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_parquet(filepath)
                print(f"   📁 {filename}: {len(df):,} rows")
                all_dataframes.append(df)
                selfplay_total += len(df)
            else:
                print(f"   ⚠️  {filename}: Not found")
        
        print(f"   🎮 Total self-play: {selfplay_total:,} rows")
        
        # Step 4: Combine all data
        print("\n🔧 Combining all datasets...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Step 5: Remove duplicates
        print("🧹 Removing duplicates...")
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates()
        after_dedup = len(combined_df)
        duplicates_removed = before_dedup - after_dedup
        
        if duplicates_removed > 0:
            print(f"   🗑️  Removed {duplicates_removed:,} duplicate rows")
        else:
            print("   ✅ No duplicates found")
        
        # Step 6: Save updated training file
        print(f"\n💾 Saving updated training file...")
        combined_df.to_parquet(self.current_train_file, index=False)
        
        # Step 7: Summary
        print("\n" + "=" * 60)
        print("📈 CONSOLIDATION SUMMARY")
        print("=" * 60)
        print(f"Original data:      {len(df_current):,} rows")
        print(f"Seed generation:    {seed_total:,} rows")
        print(f"Self-play data:     {selfplay_total:,} rows")
        print(f"Duplicates removed: {duplicates_removed:,} rows")
        print(f"Final dataset:      {len(combined_df):,} rows")
        print(f"Growth factor:      {len(combined_df) / len(df_current):.1f}x")
        print(f"Backup created:     {os.path.basename(backup_file)}")
        print("\n✅ Training data consolidation complete!")
        
        return len(combined_df), len(df_current)

    def scan_checkpoint_files(self):
        """Scan checkpoint directories for file sizes."""
        file_sizes = {}
        
        all_files = [
            (self.seed_dir, ["train_gen_code_f.parquet", "train_gen_code_i.parquet", "train_gen_code_o.parquet"]),
            (self.selfplay_dir, [
                "train_gen_code_f.parquet", "train_gen_code_i.parquet", "train_gen_code_o.parquet",
                "train_pred_code_f.parquet", "train_pred_code_i.parquet", "train_pred_code_o.parquet"
            ])
        ]
        
        for directory, filenames in all_files:
            for filename in filenames:
                filepath = os.path.join(directory, filename)
                key = f"{os.path.basename(directory)}/{filename}"
                
                if os.path.exists(filepath):
                    try:
                        df = pd.read_parquet(filepath)
                        file_sizes[key] = len(df)
                    except Exception:
                        file_sizes[key] = 0
                else:
                    file_sizes[key] = 0
        
        return file_sizes

    def detect_new_data(self, current_sizes):
        """Detect if new data has been generated."""
        if not self.last_file_sizes:
            self.last_file_sizes = current_sizes.copy()
            return 0
        
        total_new = 0
        for key, current_size in current_sizes.items():
            last_size = self.last_file_sizes.get(key, 0)
            new_samples = current_size - last_size
            if new_samples > 0:
                total_new += new_samples
        
        self.last_file_sizes = current_sizes.copy()
        return total_new

    def is_training_running(self):
        """Check if training process is still running."""
        if self.training_process:
            return self.training_process.poll() is None
        
        # Look for training process
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'main_azr_ppo.py' in cmdline:
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def stop_training(self):
        """Stop the training process gracefully."""
        if self.training_process:
            print("🛑 Stopping training process...")
            self.training_process.terminate()
            try:
                self.training_process.wait(timeout=30)
                print("✅ Training stopped gracefully")
                return True
            except subprocess.TimeoutExpired:
                print("⚠️  Force killing training process...")
                self.training_process.kill()
                return True
        return False

    def monitor_training(self):
        """Monitor training progress in background thread."""
        print(f"\n👁️  Training monitor started (check every {self.check_interval}s)")
        print(f"   Will auto-stop if stagnant for {self.stagnation_threshold}s")
        
        while self.monitoring and self.is_training_running():
            current_sizes = self.scan_checkpoint_files()
            new_data = self.detect_new_data(current_sizes)
            
            total_samples = sum(current_sizes.values())
            
            if new_data >= self.min_new_samples:
                print(f"📊 Training active: +{new_data} new samples (total: {total_samples:,})")
                self.last_activity = datetime.now()
            else:
                time_since_activity = datetime.now() - self.last_activity
                if time_since_activity.total_seconds() > self.stagnation_threshold:
                    print(f"⏰ Training stagnant for {time_since_activity}")
                    print("🛑 Auto-stopping training...")
                    
                    if self.stop_training():
                        print("✅ Training stopped successfully")
                        # Auto-consolidate after training
                        print("\n🔄 Auto-consolidating data after training completion...")
                        self.consolidate_data()
                        print("🚀 Ready for next training run!")
                    break
            
            time.sleep(self.check_interval)
        
        self.monitoring = False

    def start_training_with_monitoring(self, training_args=None):
        """Start training with automatic monitoring."""
        if not os.path.exists(self.training_script):
            print(f"❌ Training script not found: {self.training_script}")
            return False
        
        # Prepare command
        cmd = ["python", self.training_script]
        if training_args:
            cmd.extend(training_args)
        
        print(f"\n🚀 Starting AZR training with monitoring...")
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            # Start training process
            self.training_process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Start monitoring in background
            self.monitoring = True
            monitor_thread = threading.Thread(target=self.monitor_training, daemon=True)
            monitor_thread.start()
            
            # Stream training output
            print("📺 Training output:")
            print("-" * 60)
            
            for line in self.training_process.stdout:
                print(line.rstrip())
            
            # Wait for completion
            return_code = self.training_process.wait()
            self.monitoring = False
            
            if return_code == 0:
                print("\n✅ Training completed successfully!")
                return True
            else:
                print(f"\n❌ Training failed with return code {return_code}")
                return False
                
        except Exception as e:
            print(f"❌ Training error: {e}")
            self.monitoring = False
            return False

    def run_full_pipeline(self, pre_consolidate=True, training_args=None):
        """Run the complete single-GPU AZR pipeline."""
        print("🏁 AZR Single-GPU Training Pipeline")
        print("=" * 60)
        print("Hardware: RTX 5080 (99.2% cost reduction vs 8x H100)")
        print("Trade-off: ~8x longer training time for democratized access")
        print("=" * 60)
        
        # Pre-training consolidation
        if pre_consolidate:
            print("\n🔄 Phase 1: Pre-training data consolidation")
            final_rows, original_rows = self.consolidate_data()
            if final_rows > original_rows:
                print(f"🎯 Enhanced dataset ready: {final_rows:,} total samples")
        
        # Training with monitoring
        print("\n🏋️  Phase 2: Training with auto-completion monitoring")
        success = self.start_training_with_monitoring(training_args)
        
        if success:
            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("Training data consolidated and ready for next iteration!")
        else:
            print("\n💥 PIPELINE FAILED")
            print("Check logs above for details")
        
        return success

def main():
    pipeline = AZRSingleGPUPipeline()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--consolidate-only":
        # Just run consolidation
        pipeline.consolidate_data()
    else:
        # Run full pipeline
        training_args = sys.argv[1:] if len(sys.argv) > 1 else None
        pipeline.run_full_pipeline(training_args=training_args)

if __name__ == "__main__":
    main()
