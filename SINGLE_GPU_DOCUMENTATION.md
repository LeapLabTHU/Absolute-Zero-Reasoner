# AZR Single-GPU Guide: From 0.5B Experiments to 3B Training

This guide provides two distinct pipelines for running the Absolute Zero Reasoner (AZR) on consumer-grade hardware. This effort democratizes access to state-of-the-art reasoning model training, reducing the barrier to entry from an enterprise-level server farm to a single consumer GPU.

The key innovation detailed here is a stable method for integrating PyTorch FSDP with BitsAndBytes QLoRA, enabling large model training on memory-constrained hardware.

## Choosing Your Training Setup

Two fully-functional pipelines are available, each tailored for a different use case.

| Feature            | Option 1: 0.5B Automated Pipeline | Option 2: 3B QLoRA Cyclical Pipeline |
| ------------------ | ----------------------------------- | ------------------------------------------ |
| **Model Size** | 0.5 Billion parameters              | 3 Billion parameters                       |
| **Key Technology** | Standard Training + vLLM            | QLoRA + FSDP with CPU Offload + HF Rollout |
| **VRAM Usage** | ~8-12 GB (estimated)                | **16 GB** (saturates VRAM)                 |
| **System RAM** | 32 GB Recommended                   | **64 GB+** Required                        |
| **Training Speed** | Fast (~1-2 min/step)                | Slower (~14 min/step)                      |
| **Management** | `azr_0.5b_complete_automation.py`   | `master_orchestrator.py`                   |
| **Key Feature** | Auto-stops when stagnant            | Resilient to data exhaustion               |
| **Best For** | Quick experiments, learning AZR     | Deep, long-term training for SOTA results  |

---

## Technical Breakthrough: FSDP + QLoRA Integration

A primary challenge in training large models on consumer GPUs is the incompatibility between PyTorch's Fully Sharded Data Parallel (FSDP) and the 4-bit quantization from `bitsandbytes` used in QLoRA.

- **Challenge**: Standard FSDP fails when wrapping a QLoRA model, throwing a `ValueError: Cannot flatten integer dtype tensors`. This occurs because FSDP cannot handle the non-floating-point metadata tensors within the 4-bit quantized layers.

- **Solution**: The successful integration was achieved via a specific "Freeze, Inject, and Wrap" operational sequence:
    1.  **Freeze & Quantize**: The 3B base model is loaded using a `BitsAndBytesConfig`, which creates the 4-bit layers and automatically freezes them (marks them as non-trainable).
    2.  **Inject Adapters**: **Before** the FSDP wrapper is applied, the `peft` library's `get_peft_model()` function is called. This injects the small, trainable, full-precision LoRA adapter layers into the frozen model.
    3.  **Wrap Trainable Parameters**: The resulting `PeftModel` is then passed to FSDP. FSDP is configured to only shard and manage the **trainable parameters**. It therefore only "sees" the full-precision LoRA adapters and completely ignores the non-compatible quantized base layers, resolving the error.

- **Critical Insight**: This strategy bypasses the incompatibility by ensuring FSDP never has to interact with the problematic quantized tensors, making it possible to leverage QLoRA's massive memory savings in a distributed training context, even on a single GPU.

---

## Option 1: Training the 0.5B Model (Automated Pipeline)

This pipeline uses a sophisticated automation script to run the original 0.5B model. It is the ideal starting point for understanding the AZR self-play loop and for rapid experimentation.

### 🚀 Key Features
- **Rollout Engine**: Utilizes **vLLM** for optimized, high-throughput inference during the data generation phase on the non-quantized 0.5B model.
- **Fully Automated:** Runs the entire seeding-to-self-play pipeline from a single command.
- **Auto-Stopping:** Intelligently monitors data generation and automatically halts the process if training becomes stagnant.
- **Graceful Shutdown:** `Ctrl+C` safely stops the process, and `Ctrl+X` (or `kill -USR1 <pid>`) saves a checkpoint on demand without stopping.

### 🔧 Setup and Usage
1.  **Prerequisites:** An NVIDIA GPU with ~12GB VRAM, 32GB+ System RAM.
2.  **Run the Pipeline:**
    ```bash
    # To run the entire pipeline (seeding then self-play)
    python azr_0.5b_complete_automation.py --full-pipeline

    # To run only the seeding phase
    python azr_0.5b_complete_automation.py --seeding-only
    ```

---

## Option 2: Training the 3B QLoRA Model (Cyclical Orchestrator)

This is the robust, memory-efficient pipeline engineered for training the larger 3B model on a single 16GB GPU.

### 🚀 Key Features
- **Large Model on Consumer GPU:** Successfully trains a 3B model by leveraging **QLoRA** and the **FSDP+QLoRA integration** detailed above.
- **Rollout Engine**: Employs the standard **HuggingFace (`hf`) rollout** engine, which is stable and compatible with the PEFT/FSDP wrapped QLoRA model.
- **Iterative Training:** Employs a `master_orchestrator.py` to manage a stable, cyclical training loop that refines a single LoRA adapter over time.
- **Cyclical Validation:** The process is broken into 20-step training cycles. At the end of each cycle, an automated validation is run against the latest LoRA adapter checkpoint to track progress.
- **Resilient Trainer:** The core trainer code is patched to be resilient to data exhaustion (`StopIteration` errors), enabling stable, multi-day training runs.
- **Frequent & Automatic Checkpointing:** The configuration saves progress automatically every 2 steps (or as configured), minimizing potential data loss.
- **Graceful Shutdown:** The orchestrator can be stopped cleanly with `Ctrl+C`, preventing messy exits and preserving checkpoint integrity.

### 🔧 Setup and Usage
1.  **Prerequisites:** An NVIDIA GPU with **16GB VRAM**, **64GB+ System RAM**.
2.  **Important Dependencies**:
    > This setup was stabilized on a specific software stack. For successful replication, please note:
    > * **PyTorch Version:** A nightly build of PyTorch **(v2.8.0.dev for CUDA 12.8)** was used.
    > * **Build from Source:** Key libraries such as **`flash-attn`** and **`xformers`** may need to be built from source against your specific PyTorch and CUDA versions.
3.  **Run the Pipeline:**
    ```bash
    # This command manages the entire cyclical training process
    python master_orchestrator.py
    ```
### 📊 Performance Reality
- **Training Speed:** Approx. **850 seconds / 14 minutes** per step.
- **1000 Steps Estimate:** ~236 hours / ~10 days.
- **Memory Profile:** Saturates **16GB VRAM** and uses **~42GB of System RAM** for offloading.

This setup represents a significant trade-off: sacrificing time to gain a **~99% reduction in hardware cost**, making 3B-scale reasoning research accessible.