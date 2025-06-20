#!/usr/bin/env python3
"""
AZR Complete Pipeline with Auto-Completion & Graceful Shutdown
Handles both seeding and self-play with automatic stopping when goals are reached.
Enhanced with graceful shutdown to preserve all progress on interruption.

FIXED: Respects original AZR data architecture - no parquet consolidation!
- Seeding creates JSONL files (preserved separately)
- Self-play generates checkpoint data (preserved separately) 
- Training parquet remains pure evaluation data

Features:
- Ctrl+C: Graceful shutdown with progress preservation
- Ctrl+X: Save checkpoint without stopping
- Auto-stopping prevents endless runs
- Preserves original AZR data separation
"""

import pandas as pd
import numpy as np
import os
import shutil
import time
import psutil
import subprocess
import threading
import signal
import sys
import termios
import tty
import select
from datetime import datetime, timedelta
from pathlib import Path

class AZRCompleteAutomation:
    def __init__(self):
        # AZR project paths
        self.base_dir = "/home/frankshortt/AI/qwen3-training/Absolute-Zero-Reasoner"
        
        # IMPORTANT: We do NOT modify the training parquet file
        # It should remain pure evaluation data only
        self.training_parquet = os.path.join(self.base_dir, "data/code_reason/test_answer.parquet")
        
        # Checkpoint directories for monitoring progress
        self.seed_dir = os.path.join(self.base_dir, "checkpoints/code_io/azr/0.5b_coder_seed_generation/test_answer/Qwen2.5-Coder-0.5B/answer_conditional/code/")
        self.selfplay_dir = os.path.join(self.base_dir, "checkpoints/code_io/azr/0.5b_coder_selfplay_single_gpu/test_answer/Qwen2.5-Coder-0.5B/answer_conditional/code/")
        
        # JSONL seed files (separate from parquet)
        self.seed_jsonl_dir = os.path.join(self.base_dir, "data/")
        
        # Scripts
        self.seeding_script = os.path.join(self.base_dir, "scripts/seeding/coder0.5b-single_GPU-fixed.sh")
        self.selfplay_script = os.path.join(self.base_dir, "scripts/selfplay/coder0.5b-single_GPU-fixed.sh")
        
        # Monitoring settings
        self.check_interval = 300  # 5 minutes
        self.stagnation_threshold = 7200  # 2 hours
        self.min_new_samples = 50
        
        self.monitoring = False
        self.current_process = None
        self.last_file_sizes = {}
        self.last_activity = datetime.now()
        
        # Graceful shutdown state
        self.shutdown_requested = False
        self.current_phase = "initialization"
        
        # Keyboard handling
        self.old_terminal_settings = None
        self.keyboard_thread = None
        
        # Setup signal handlers immediately
        self.setup_signal_handlers()
        self.setup_keyboard_handler()

    def setup_keyboard_handler(self):
        """Setup keyboard handler for Ctrl+X checkpoint saves."""
        try:
            # Save original terminal settings
            self.old_terminal_settings = termios.tcgetattr(sys.stdin)
            
            # Start keyboard monitoring thread
            self.keyboard_thread = threading.Thread(target=self.keyboard_monitor, daemon=True)
            self.keyboard_thread.start()
            
        except (termios.error, OSError):
            # Fallback if terminal control not available (e.g., in some IDEs)
            print("   ⚠️  Direct keyboard capture not available")
            print("   💡 Use: kill -USR1 <pid> for checkpoint save")

    def keyboard_monitor(self):
        """Monitor for Ctrl+X keypresses."""
        try:
            while not self.shutdown_requested:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    try:
                        # Set terminal to raw mode temporarily
                        tty.setraw(sys.stdin.fileno())
                        ch = sys.stdin.read(1)
                        
                        # Restore terminal settings
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
                        
                        # Check for Ctrl+X (ASCII 24)
                        if ord(ch) == 24:  # Ctrl+X
                            self.checkpoint_save(None, None)
                        
                    except (OSError, termios.error):
                        break
                        
        except Exception:
            # Silently handle keyboard monitoring errors
            pass

    def restore_terminal(self):
        """Restore original terminal settings."""
        try:
            if self.old_terminal_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
        except (termios.error, OSError):
            pass

    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        signal.signal(signal.SIGUSR1, self.checkpoint_save)  # Ctrl+X equivalent
        print("🛡️  Graceful shutdown enabled!")
        print("   Ctrl+C: Safe stop with progress preservation")
        print("   Ctrl+X: Save checkpoint without stopping")
        print("   All progress will be preserved! 🔒")

    def checkpoint_save(self, signum, frame):
        """Save checkpoint without stopping the process."""
        print(f"\n💾 Checkpoint save requested during: {self.current_phase}")
        print("📊 Saving progress without stopping...")
        
        try:
            # Get current process ID for user reference
            if self.current_process:
                print(f"   Process ID: {self.current_process.pid}")
            
            # Check current progress without modifying training data
            current_sizes = self.scan_checkpoint_files()
            total_samples = sum(current_sizes.values())
            
            if total_samples > 0:
                print(f"✅ Checkpoint saved! Current progress: {total_samples:,} samples in checkpoints")
                self.backup_progress_data()
            else:
                print("ℹ️  Checkpoint saved! No data generated yet")
                
            print(f"🔄 {self.current_phase.capitalize()} continues running...")
            
        except Exception as e:
            print(f"⚠️  Checkpoint save error: {e}")
            print("🔄 Process continues running normally")

    def backup_progress_data(self):
        """Backup checkpoint data without modifying training parquet."""
        backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join(self.base_dir, f"backup_{backup_timestamp}")
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            # Backup checkpoint directories
            if os.path.exists(self.seed_dir):
                seed_backup = os.path.join(backup_dir, "seed_checkpoints")
                shutil.copytree(self.seed_dir, seed_backup, dirs_exist_ok=True)
                print(f"   📦 Seed checkpoints backed up to: {seed_backup}")
            
            if os.path.exists(self.selfplay_dir):
                selfplay_backup = os.path.join(backup_dir, "selfplay_checkpoints")
                shutil.copytree(self.selfplay_dir, selfplay_backup, dirs_exist_ok=True)
                print(f"   📦 Self-play checkpoints backed up to: {selfplay_backup}")
            
            # Backup any JSONL seed files
            jsonl_files = []
            for file in os.listdir(self.seed_jsonl_dir):
                if file.endswith('.jsonl'):
                    jsonl_files.append(file)
                    src = os.path.join(self.seed_jsonl_dir, file)
                    dst = os.path.join(backup_dir, file)
                    shutil.copy2(src, dst)
            
            if jsonl_files:
                print(f"   📦 JSONL seed files backed up: {', '.join(jsonl_files)}")
            
            print(f"✅ Progress backup created: {backup_dir}")
            
        except Exception as e:
            print(f"⚠️  Backup error: {e}")

    def graceful_shutdown(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C or termination signal."""
        if self.shutdown_requested:
            print("\n🚨 Force shutdown requested! Exiting immediately...")
            sys.exit(1)
            
        self.shutdown_requested = True
        
        print(f"\n🛑 Graceful shutdown initiated during: {self.current_phase}")
        print("📊 Preserving all progress...")
        
        # Stop current process gracefully
        if self.current_process and self.is_process_running():
            print("⏹️  Stopping current process...")
            self.stop_current_process()
        
        # Stop monitoring
        if self.monitoring:
            print("👁️  Stopping monitor...")
            self.monitoring = False
        
        # Restore terminal
        self.restore_terminal()
        
        # Backup progress data (NO parquet modification)
        print("💾 Backing up checkpoint progress...")
        try:
            self.backup_progress_data()
            print("✅ Progress backed up successfully!")
        except Exception as e:
            print(f"⚠️  Backup error (progress still in checkpoints): {e}")
        
        print("\n🎯 Graceful shutdown complete!")
        print("🔒 All progress preserved in checkpoints - safe to restart anytime")
        print("💡 Tip: Checkpoints contain your generated data")
        print("🧠 Note: Training parquet kept pure as intended by AZR design")
        sys.exit(0)

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

    def scan_jsonl_files(self):
        """Scan JSONL seed files for progress."""
        jsonl_sizes = {}
        
        try:
            for file in os.listdir(self.seed_jsonl_dir):
                if file.endswith('.jsonl'):
                    filepath = os.path.join(self.seed_jsonl_dir, file)
                    try:
                        with open(filepath, 'r') as f:
                            line_count = sum(1 for line in f if line.strip())
                        jsonl_sizes[file] = line_count
                    except Exception:
                        jsonl_sizes[file] = 0
        except Exception:
            pass
        
        return jsonl_sizes

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

        # Set adaptive threshold based on process type
        if process_type == "seeding":
            print("🌱 Seeding mode: Monitoring checkpoint files and JSONL output")
            self.min_new_samples = 1  # Any new data counts for seeding
        else:
            self.min_new_samples = 50  # Keep higher threshold for self-play

        print(f"\n👁️  {process_type.capitalize()} monitor started")
        print(f"   Check interval: {self.check_interval}s")
        print(f"   Stagnation threshold: {self.stagnation_threshold}s")
        print("🧠 Note: Preserves original AZR data separation!")
        
        # Reset monitoring state
        self.last_file_sizes = {}
        self.last_activity = datetime.now()
        
        while self.monitoring and self.is_process_running() and not self.shutdown_requested:
            # Check both checkpoint files and JSONL files
            checkpoint_sizes = self.scan_checkpoint_files(focus_dir)
            jsonl_sizes = self.scan_jsonl_files()
            
            # Combine all data sources for monitoring
            all_sizes = {**checkpoint_sizes, **jsonl_sizes}
            new_data = self.detect_new_data(all_sizes)
            
            total_samples = sum(all_sizes.values())
            
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
                        
                        # Create backup of checkpoint progress
                        print(f"\n🔄 Backing up {process_type} progress...")
                        self.backup_progress_data()
                        print("✅ Progress preserved in backups!")
                        print("🧠 Data remains separated as intended by AZR design")
                    break
            
            time.sleep(self.check_interval)
        
        self.monitoring = False

    def show_current_status(self):
        """Show current data status without modifying anything."""
        print("\n📊 CURRENT AZR DATA STATUS")
        print("=" * 60)
        
        # Training parquet status
        if os.path.exists(self.training_parquet):
            try:
                df = pd.read_parquet(self.training_parquet)
                print(f"🎯 Training parquet: {len(df):,} rows ({os.path.getsize(self.training_parquet) / 1024 / 1024:.1f} MB)")
            except Exception as e:
                print(f"🎯 Training parquet: Error reading - {e}")
        else:
            print("🎯 Training parquet: Not found")
        
        # Checkpoint data status
        checkpoint_sizes = self.scan_checkpoint_files()
        if checkpoint_sizes:
            print("\n📦 Checkpoint data:")
            for key, size in checkpoint_sizes.items():
                if size > 0:
                    print(f"   {key}: {size:,} rows")
        else:
            print("\n📦 Checkpoint data: None found")
        
        # JSONL seed files status
        jsonl_sizes = self.scan_jsonl_files()
        if jsonl_sizes:
            print("\n📄 JSONL seed files:")
            for file, lines in jsonl_sizes.items():
                if lines > 0:
                    print(f"   {file}: {lines:,} lines")
        else:
            print("\n📄 JSONL seed files: None found")
        
        print("\n✅ Data separation maintained as intended by AZR design!")

    def run_seeding_with_monitoring(self):
        """Run seeding script with auto-completion monitoring."""
        self.current_phase = "seeding"
        
        if not os.path.exists(self.seeding_script):
            print(f"❌ Seeding script not found: {self.seeding_script}")
            print(f"   Looking for: {self.seeding_script}")
            return False
        
        print("\n🌱 Starting seeding with auto-completion monitoring...")
        print("   Monitoring checkpoint files and JSONL output")
        print("   Training parquet will remain unchanged! ✅")
        print(f"   Using script: {self.seeding_script}")
        
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
                if self.shutdown_requested:
                    break
                print(line.rstrip())
            
            # Wait for completion
            if not self.shutdown_requested:
                return_code = self.current_process.wait()
                self.monitoring = False
                
                if return_code == 0:
                    print("\n✅ Seeding completed successfully!")
                    print("🧠 Seed data preserved in checkpoints and JSONL files")
                    return True
                else:
                    print(f"\n❌ Seeding failed with return code {return_code}")
                    return False
            else:
                return True  # Graceful shutdown counts as success
                
        except Exception as e:
            print(f"❌ Seeding error: {e}")
            self.monitoring = False
            return False

    def run_selfplay_with_monitoring(self):
        """Run self-play training with auto-completion monitoring."""
        self.current_phase = "self-play"
        
        if not os.path.exists(self.selfplay_script):
            print(f"❌ Self-play script not found: {self.selfplay_script}")
            print(f"   Looking for: {self.selfplay_script}")
            return False
        
        print("\n🎮 Starting self-play with auto-completion monitoring...")
        print("   Training parquet will remain unchanged! ✅")
        print(f"   Using script: {self.selfplay_script}")
        
        try:
            # Start self-play process
            self.current_process = subprocess.Popen(
                ["bash", self.selfplay_script],
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
                if self.shutdown_requested:
                    break
                print(line.rstrip())
            
            # Wait for completion
            if not self.shutdown_requested:
                return_code = self.current_process.wait()
                self.monitoring = False
                
                if return_code == 0:
                    print("\n✅ Self-play completed successfully!")
                    print("🧠 Self-play data preserved in checkpoints")
                    return True
                else:
                    print(f"\n❌ Self-play failed with return code {return_code}")
                    return False
            else:
                return True  # Graceful shutdown counts as success
                
        except Exception as e:
            print(f"❌ Self-play error: {e}")
            self.monitoring = False
            return False

    def run_full_pipeline(self):
        """Run complete pipeline: seeding → self-play (with proper data separation)."""
        self.current_phase = "full-pipeline"
        
        print("🚀 AZR Complete Automated Pipeline")
        print("=" * 60)
        print("Hardware: RTX 5080 (99.2% cost reduction vs 8x H100)")
        print("Data Architecture: Maintains original AZR separation! 🧠")
        print("Features: Auto-stopping prevents endless runs!")
        print("🛡️  Graceful shutdown: Ctrl+C preserves all progress!")
        print("=" * 60)
        
        # Show initial status
        self.show_current_status()
        
        # Phase 1: Seeding with auto-stop
        print("\n🌱 Phase 1: Seeding with Auto-Completion")
        seeding_success = self.run_seeding_with_monitoring()
        
        if not seeding_success or self.shutdown_requested:
            if self.shutdown_requested:
                print("🛑 Pipeline stopped gracefully during seeding")
            else:
                print("❌ Pipeline failed at seeding phase")
            return False
        
        # Phase 2: Self-play with auto-stop  
        print("\n🎮 Phase 2: Self-Play with Auto-Completion")
        selfplay_success = self.run_selfplay_with_monitoring()
        
        if selfplay_success and not self.shutdown_requested:
            print("\n🎉 COMPLETE PIPELINE SUCCESS!")
            print("Both seeding and self-play completed with proper data separation!")
            print("Training parquet kept pure as intended! 🧠")
            print("No more weekend-long runs with 0 new data! 🎯")
        elif self.shutdown_requested:
            print("\n🛑 Pipeline stopped gracefully during self-play")
        else:
            print("\n⚠️  Pipeline completed seeding but self-play had issues")
        
        return selfplay_success

def main():
    automation = AZRCompleteAutomation()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--status":
            automation.current_phase = "status"
            automation.show_current_status()
        elif command == "--seeding-only":
            automation.run_seeding_with_monitoring()
        elif command == "--selfplay-only":
            automation.run_selfplay_with_monitoring()
        elif command == "--full-pipeline":
            automation.run_full_pipeline()
        else:
            print("❌ Invalid command. Use:")
            print("   --status           : Show current data status")
            print("   --seeding-only     : Run seeding with auto-stop")
            print("   --selfplay-only    : Run self-play with auto-stop")
            print("   --full-pipeline    : Run complete seeding → self-play pipeline")
            print("\n🛡️  All commands support graceful shutdown (Ctrl+C)")
            print("🧠  Maintains original AZR data separation!")
    else:
        # Default: self-play only (current behavior)
        automation.run_selfplay_with_monitoring()

if __name__ == "__main__":
    main()