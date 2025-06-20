#!/usr/bin/env python3
"""
Advanced FSDP + QLoRA Model Consolidator

This script manually merges a trained LoRA adapter from a QLoRA+FSDP checkpoint
into a 4-bit quantized base model, de-quantizes the result, and saves a final,
full-precision Hugging Face model.

This replaces the multi-step merge/consolidate process.
"""

import torch
import argparse
import os
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig
from peft.utils import get_peft_model_state_dict
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_lora_config(model_name="Qwen/Qwen2.5-Coder-3B"):
    """Returns the specific LoRA config used during training."""
    from peft import LoraConfig, TaskType
    return LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

def main(fsdp_checkpoint_path, output_dir, model_name="Qwen/Qwen2.5-Coder-3B"):
    logging.info("--- 🚀 Advanced FSDP+QLoRA Consolidation Starting ---")
    
    # --- 1. Load the Base Model in 4-bit ---
    # We must load the model in the *exact* same quantization config as training
    # to ensure the base layer weights match the format in the checkpoint.
    logging.info(f"Loading base model '{model_name}' with 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="cpu", # Load to CPU to manually manage layers
    )
    
    # --- 2. Inject LoRA Adapters to Create Structure ---
    # We add new, untrained LoRA adapters to the model. This gives the model
    # the 'lora_A' and 'lora_B' layers we need to load our trained weights into.
    logging.info("Injecting empty LoRA adapters into the base model...")
    from peft import get_peft_model
    lora_config = get_lora_config(model_name)
    model = get_peft_model(model, lora_config)
    
    # --- 3. Load the Trained LoRA Weights ---
    # The FSDP checkpoint contains *only* the trained LoRA adapter weights,
    # because the base model was frozen.
    state_dict_path = os.path.join(fsdp_checkpoint_path, "model_world_size_1_rank_0.pt")
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"FSDP state dict not found at: {state_dict_path}")
        
    logging.info(f"Loading trained weights from FSDP checkpoint: {state_dict_path}")
    fsdp_state_dict = torch.load(state_dict_path, map_location="cpu")
    
    # We only need the LoRA weights from this checkpoint.
    lora_state_dict = get_peft_model_state_dict(model, state_dict=fsdp_state_dict)

    model.load_state_dict(lora_state_dict, strict=False)
    logging.info("✅ Successfully loaded trained LoRA weights into adapter layers.")

    # --- 4. Merge and De-quantize ---
    # This is the critical step. `merge_and_unload` combines the LoRA weights
    # into the 4-bit base layers and then de-quantizes everything back to full precision.
    logging.info("Merging LoRA adapters and de-quantizing model to full precision...")
    model = model.merge_and_unload()
    logging.info("✅ Model successfully merged and de-quantized.")

    # --- 5. Save the Final, Full-Precision Model ---
    logging.info(f"Saving final consolidated model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    
    # Save the tokenizer for a complete Hugging Face model directory
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    
    logging.info("--- 🎉 Consolidation Complete! ---")
    logging.info(f"Final, full-precision model is ready at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate a QLoRA+FSDP checkpoint.")
    parser.add_argument("--fsdp_checkpoint_path", type=str, required=True, help="Path to the 'actor' directory of the FSDP checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the final consolidated Hugging Face model.")
    args = parser.parse_args()
    
    main(args.fsdp_checkpoint_path, args.output_dir)
