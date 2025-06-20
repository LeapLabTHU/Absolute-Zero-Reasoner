#!/usr/bin/env python3
"""
Simple and Correct Cyclical Training Orchestrator - V3 FINAL

- Implements robust, memory-safe cyclical training by leveraging the training
  framework's built-in resume capability to iteratively train ONE adapter.
- Validates the LoRA checkpoint directly after each cycle.
- Consolidates only ONCE at the very end.
"""

import subprocess
import sys
import os
import glob
import re

# --- CONFIGURATION ---
BASE_MODEL_PATH = "Qwen/Qwen2.5-Coder-3B"
TOTAL_CYCLES = 20
STEPS_PER_CYCLE = 20
CHECKPOINT_ROOT = "checkpoints/code_io/azr/3b_coder_training_optimized/test_answer/Qwen2.5-Coder-3B/answer_conditional"
TRAINING_SCRIPT_PATH = "/home/frankshortt/AI/qwen3-training/Absolute-Zero-Reasoner/scripts/selfplay/coder3b-single_GPU.sh"
VALIDATION_SCRIPT_PATH = "/home/frankshortt/AI/qwen3-training/Absolute-Zero-Reasoner/validate_lora_checkpoint.py"
FINAL_CONSOLIDATION_SCRIPT_PATH = "/home/frankshortt/AI/qwen3-training/Absolute-Zero-Reasoner/consolidate_fsdp_qlora.py"
VALIDATION_FILE_PATH = "/home/frankshortt/AI/qwen3-training/Absolute-Zero-Reasoner/data/code_reason/test_answer.parquet"


def run_command(command: str, env=None):
    """Runs a command, streams output, and checks for errors."""
    print(f"\n--- EXECUTING COMMAND ---\n{command}\n-------------------------\n")
    process = subprocess.Popen(
        command, shell=True, executable='/bin/bash', env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
    )
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        print(f"\n### ERROR: Command failed with return code {return_code}. Aborting. ###")
        sys.exit(1)

def find_latest_step(checkpoint_root):
    """Finds the highest step number from checkpoint directories."""
    pattern = os.path.join(checkpoint_root, "global_step_*")
    checkpoint_dirs = glob.glob(pattern)
    if not checkpoint_dirs:
        return 0
    latest_step = 0
    for path in checkpoint_dirs:
        match = re.search(r'global_step_(\d+)', path)
        if match:
            latest_step = max(latest_step, int(match.group(1)))
    return latest_step

def main():
    """The main orchestration loop."""
    for cycle_num in range(1, TOTAL_CYCLES + 1):
        print(f"\n========================================================")
        print(f"--- 🚀 STARTING TRAINING CYCLE {cycle_num} / {TOTAL_CYCLES} 🚀")

        latest_step = find_latest_step(CHECKPOINT_ROOT)
        target_step = cycle_num * STEPS_PER_CYCLE

        if latest_step >= target_step:
            print(f"--- Cycle {cycle_num} already complete (found step {latest_step} >= target {target_step}). Skipping.")
        else:
            print(f"--- Training from step {latest_step} up to {target_step}")
            
            env = os.environ.copy()
            env["AZR_MODEL_PATH"] = BASE_MODEL_PATH
            env["AZR_TARGET_STEPS"] = str(target_step)
            env["AZR_CYCLE_NUM"] = str(cycle_num)

            # ======================== START: THE BRUTE FORCE FIX ========================
            # Explicitly put your project directory at the front of the Python path.
            # This forces Python to look here first for any imports.
            project_root = "/home/frankshortt/AI/qwen3-training/Absolute-Zero-Reasoner"
            env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
            # print(f"--- FORCE-INJECTING PYTHONPATH: {env['PYTHONPATH']}")
            # ========================= END: THE BRUTE FORCE FIX =========================

            run_command(f"bash {TRAINING_SCRIPT_PATH}", env=env)

        print(f"\n--- VALIDATING CHECKPOINT FOR CYCLE {cycle_num} ---")
        lora_checkpoint_path = os.path.join(CHECKPOINT_ROOT, f"global_step_{target_step}", "actor")

        if os.path.exists(lora_checkpoint_path) and os.path.exists(VALIDATION_SCRIPT_PATH):
             run_command(
                f"python {VALIDATION_SCRIPT_PATH} "
                f"--base_model '{BASE_MODEL_PATH}' "
                f"--lora_checkpoint '{lora_checkpoint_path}' "
                f"--validation_file '{VALIDATION_FILE_PATH}'"
            )
        else:
            print(f"⚠️ Warning: Cannot validate. Checkpoint or validation script not found.")
            print(f"   - Searched for checkpoint: {lora_checkpoint_path}")
            print(f"   - Searched for script: {VALIDATION_SCRIPT_PATH}")


        print(f"\n--- ✅ CYCLE {cycle_num} COMPLETE. Adapter checkpoint at: {lora_checkpoint_path} ---")

    print("\n========================================================")
    print("--- 🎉 ALL CYCLES COMPLETE - PERFORMING FINAL CONSOLIDATION 🎉 ---")
    final_step = TOTAL_CYCLES * STEPS_PER_CYCLE
    final_lora_path = os.path.join(CHECKPOINT_ROOT, f"global_step_{final_step}", "actor")
    final_output_path = "models/final_consolidated_model"

    if os.path.exists(final_lora_path) and os.path.exists(FINAL_CONSOLIDATION_SCRIPT_PATH):
        run_command(
            f"python {FINAL_CONSOLIDATION_SCRIPT_PATH} "
            f"--fsdp_checkpoint_path '{final_lora_path}' "
            f"--output_dir '{final_output_path}'"
        )
        print(f"\n--- ✅ Final consolidated model saved to: {final_output_path} ---")
    else:
        print(f"⚠️ Error: Final checkpoint for consolidation not found at {final_lora_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Graceful shutdown initiated by user. Orchestrator loop terminated.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n💥 An unexpected error occurred in the orchestrator: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)