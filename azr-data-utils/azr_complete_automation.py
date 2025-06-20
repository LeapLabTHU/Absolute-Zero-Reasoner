#!/usr/bin/env python3
"""
AZR Complete Pipeline with Auto-Completion
Handles both seeding and self-play with automatic stopping when goals are reached.

Features:
- Seeding with auto-stop (no more weekend-long runs with 0 new data)
- Self-play with auto-stop 
- Automatic data consolidation between phases
- Full pipeline or individual phase execution
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

class AZRCompleteAutomation:
    def __init__(self):
        # AZR project paths
        self.base_dir = "/home/frankshortt/AI/qwen3-training/Absolute-Zero-Reasoner"
        self.current_train_file = os.path.join(self.base_dir, "data/code_reason/test_answer.parquet")
        
        # Checkpoint directories  
        self.seed_dir = os.path.join(self.base_dir, "checkpoints/code_io/azr/0.5b_coder_seed_generation/test_answer/Qwen2.5-Coder-0.5B/answer_conditional/code/")
        self.selfplay_dir = os.path.join(self.base_dir, "checkpoints/code_io/azr/0.5b_coder_selfplay_single_gpu/test_answer/Qwen2.5-Coder-0.5B/answer_conditional/code/")
        
        # Scripts
        self.seeding_script = os.path.join(self.base_dir, "coder0.5b-single_GPU-fixed.sh")
        self.training_script = os.path.join(self.base_dir, "main_azr_ppo.py")
        
        # Monitoring settings
        self.check_interval = 300  # 5 minutes
        self.stagnation_threshold = 1800  # 30 minutes
        self.min_new_samples = 10
        
        self.monitoring = False
        self.current_process = None
        self.last_file_sizes = {}
        self.last_activity = datetime.now()

    def create_backup(self):
        """Create backup of current training file."""
        backup_file = os.path.join(
            self.base_dir, 
            f"data/code_reason/test_answer_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        )
        if os.path.exists(self.current_train_file):
            print(f"📦 Creating backup: {os.path.basename(backup_file)}")
            shutil.copy2(self.current_train_file, backup_file)
            return backup_file
        return None

    def consolidate_data(self):
        """Enhanced data consolidation with both seed and self-play data."""
        print("🔍 AZR Data Consolidation Starting...")
        print("=" * 60)
        
        # Create backup
        backup_file = self.create_backup()
        
        # Load current training data
        print("📊 Loading current training data...")
        if os.path.exists(self.current_train_file):
            df_current = pd.read_parquet(self.current_train_file)
            print(f"   Current training data: {len(df_current):,} rows")
        else:
            print("   No existing training data found, starting fresh")
            df_current = pd.DataFrame()
        
        all_dataframes = [df_current] if not df_current.empty else []
        
        # Process seed generation data
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
        
        # Process self-play data
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
        
        if not all_dataframes:
            print("❌ No data found to consolidate!")
            return 0, 0
        
        # Combine all data
        print("\n🔧 Combining all datasets...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Remove duplicates
        print("🧹 Removing duplicates...")
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates()
        after_dedup = len(combined_df)
        duplicates_removed = before_dedup - after_dedup
        
        if duplicates_removed > 0:
            print(f"   🗑️  Removed {duplicates_removed:,} duplicate rows")
        else:
            print("   ✅ No duplicates found")
        
        # Save updated training file
        print(f"\n💾 Saving updated training file...")
        os.makedirs(os.path.dirname(self.current_train_file), exist_ok=True)
        combined_df.to_parquet(self.current_train_file, index=False)
        
        # Summary
        print("\n" + "=" * 60)
        print("📈 CONSOLIDATION SUMMARY")
        print("=" * 60)
        original_count = len(df_current) if not df_current.empty else 0
        print(f"Original data:      {original_count:,} rows")
        print(f"Seed generation:    {seed_total:,} rows")
        print(f"Self-play data:     {selfplay_total:,} rows")
        print(f"Duplicates removed: {duplicates_removed:,} rows")
        print(f"Final dataset:      {len(combined_df):,} rows")
        if original_count > 0:
            print(f"Growth factor:      {len(combined_df) / original_count:.1f}x")
        if backup_file:
            print(f"Backup created:     {os.path.basename(backup_file)}")
        print("\n✅ Training data consolidation complete!")
        
        return len(combined_df), original_count

    def scan_checkpoint_files(self, focus_dir=None):
        """Scan checkpoint directories for new data."""
        file_sizes = {}
        
        # Define directories to scan
        if focus_dir == "seed":
            scan_dirs = [(self.seed_dir, ["train_gen_code_f.parquet", "train_gen_code_i.parquet", "train_gen_code_o.parquet"])]
        elif focus_dir == "selfplay":
            scan_dirs = [(self.selfplay_dir, [
                "train_gen_code_f.parquet", "train_gen_code_i.parquet", "train_gen_code_o.parquet",
                "train_pred_code_f.parquet", "train_pred_code_i.parquet", "train_pred_code_o.parquet"
            ])]
        else:
            # Scan both
            scan_dirs = [
                (self.seed_dir, ["train_gen_code_f.parquet", "train_gen_code_i.parquet", "train_gen_code_o.parquet"]),
                (self.selfplay_dir, [
                    "train_gen_code_f.parquet", "train_gen_code_i.parquet", "train_gen_code_o.parquet",
                    "train_pred_code_f.parquet", "train_pred_code_i.parquet", "train_pred_code_o.parquet"
                ])
            ]
        
        for directory, filenames in scan_dirs:
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
        new_details = []
        
        for key, current_size in current_sizes.items():
            last_size = self.last_file_sizes.get(key, 0)
            new_samples = current_size - last_size
            if new_samples > 0:
                total_new += new_samples
                new_details.append(f"{key}: +{new_samples}")
        
        if new_details:
            print(f"   📊 New data detected: {', '.join(new_details)}")
        
        self.last_file_sizes = current_sizes.copy()
        return total_new

    def is_process_running(self):
        """Check if current process is still running."""
        if self.current_process:
            return self.current_process.poll() is None
        return False

    def stop_current_process(self):
        """Stop the current process gracefully."""
        if self.current_process:
            print("🛑 Stopping current process...")
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=30)
                print("✅ Process stopped gracefully")
                return True
            except subprocess.TimeoutExpired:
                print("⚠️  Force killing process...")
                self.current_process.kill()
                return True
        return False

    def monitor_process(self, process_type="training", focus_dir=None):
        """Monitor process progress and auto-stop when stagnant."""
        print(f"\n👁️  {process_type.capitalize()} monitor started")
        print(f"   Check interval: {self.check_interval}s")
        print(f"   Stagnation threshold: {self.stagnation_threshold}s")
        
        # Reset monitoring state
        self.last_file_sizes = {}
        self.last_activity = datetime.now()
        
        while self.monitoring and self.is_process_running():
            current_sizes = self.scan_checkpoint_files(focus_dir)
            new_data = self.detect_new_data(current_sizes)
            
            total_samples = sum(current_sizes.values())
            
            if new_data >= self.min_new_samples:
                print(f"✅ {process_type.capitalize()} active: +{new_data} new samples (total: {total_samples:,})")
                self.last_activity = datetime.now()
            else:
                time_since_activity = datetime.now() - self.last_activity
                print(f"⏳ {process_type.capitalize()} stagnant for {time_since_activity} (total: {total_samples:,})")
                
                if time_since_activity.total_seconds() > self.stagnation_threshold:
                    print(f"🎯 {process_type.capitalize()} completed! Auto-stopping...")
                    
                    if self.stop_current_process():
                        print(f"✅ {process_type.capitalize()} stopped successfully")
                        
                        # Auto-consolidate after completion
                        print(f"\n🔄 Auto-consolidating data after {process_type} completion...")
                        self.consolidate_data()
                        print("🚀 Data ready for next phase!")
                    break
            
            time.sleep(self.check_interval)
        
        self.monitoring = False

    def run_seeding_with_monitoring(self):
        """Run seeding script with auto-completion monitoring."""
        if not os.path.exists(self.seeding_script):
            print(f"❌ Seeding script not found: {self.seeding_script}")
            return False
        
        print("\n🌱 Starting seeding with auto-completion monitoring...")
        print("   This will prevent endless seeding runs!")
        
        try:
            # Start seeding process
            self.current_process = subprocess.Popen(
                ["bash", self.seeding_script],
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Start monitoring in background
            self.monitoring = True
            monitor_thread = threading.Thread(
                target=lambda: self.monitor_process("seeding", "seed"), 
                daemon=True
            )
            monitor_thread.start()
            
            # Stream output
            print("📺 Seeding output:")
            print("-" * 60)
            
            for line in self.current_process.stdout:
                print(line.rstrip())
            
            # Wait for completion
            return_code = self.current_process.wait()
            self.monitoring = False
            
            if return_code == 0:
                print("\n✅ Seeding completed successfully!")
                return True
            else:
                print(f"\n❌ Seeding failed with return code {return_code}")
                return False
                
        except Exception as e:
            print(f"❌ Seeding error: {e}")
            self.monitoring = False
            return False

    def run_selfplay_with_monitoring(self):
        """Run self-play training with auto-completion monitoring."""
        if not os.path.exists(self.training_script):
            print(f"❌ Training script not found: {self.training_script}")
            return False
        
        print("\n🎮 Starting self-play with auto-completion monitoring...")
        
        # Basic self-play command (user can customize)
        cmd = ["python", self.training_script]
        
        try:
            # Start self-play process
            self.current_process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Start monitoring
            self.monitoring = True
            monitor_thread = threading.Thread(
                target=lambda: self.monitor_process("self-play", "selfplay"), 
                daemon=True
            )
            monitor_thread.start()
            
            # Stream output
            print("📺 Self-play output:")
            print("-" * 60)
            
            for line in self.current_process.stdout:
                print(line.rstrip())
            
            # Wait for completion
            return_code = self.current_process.wait()
            self.monitoring = False
            
            if return_code == 0:
                print("\n✅ Self-play completed successfully!")
                return True
            else:
                print(f"\n❌ Self-play failed with return code {return_code}")
                return False
                
        except Exception as e:
            print(f"❌ Self-play error: {e}")
            self.monitoring = False
            return False

    def run_full_pipeline(self):
        """Run complete pipeline: seeding → merge → self-play → merge."""
        print("🚀 AZR Complete Automated Pipeline")
        print("=" * 60)
        print("Hardware: RTX 5080 (99.2% cost reduction vs 8x H100)")
        print("Trade-off: ~8x longer training time for democratized access")
        print("Features: Auto-stopping prevents endless runs!")
        print("=" * 60)
        
        # Phase 1: Seeding with auto-stop
        print("\n🌱 Phase 1: Seeding with Auto-Completion")
        seeding_success = self.run_seeding_with_monitoring()
        
        if not seeding_success:
            print("❌ Pipeline failed at seeding phase")
            return False
        
        # Phase 2: Self-play with auto-stop  
        print("\n🎮 Phase 2: Self-Play with Auto-Completion")
        selfplay_success = self.run_selfplay_with_monitoring()
        
        if selfplay_success:
            print("\n🎉 COMPLETE PIPELINE SUCCESS!")
            print("Both seeding and self-play completed with auto-stopping!")
            print("No more weekend-long runs with 0 new data! 🎯")
        else:
            print("\n⚠️  Pipeline completed seeding but self-play had issues")
        
        return selfplay_success

def main():
    automation = AZRCompleteAutomation()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--consolidate-only":
            automation.consolidate_data()
        elif command == "--seeding-only":
            automation.run_seeding_with_monitoring()
        elif command == "--selfplay-only":
            automation.run_selfplay_with_monitoring()
        elif command == "--full-pipeline":
            automation.run_full_pipeline()
        else:
            print("❌ Invalid command. Use:")
            print("   --consolidate-only  : Just merge existing data")
            print("   --seeding-only      : Run seeding with auto-stop")
            print("   --selfplay-only     : Run self-play with auto-stop")
            print("   --full-pipeline     : Run complete seeding → self-play pipeline")
    else:
        # Default: self-play only (current behavior)
        automation.run_selfplay_with_monitoring()

if __name__ == "__main__":
    main()
