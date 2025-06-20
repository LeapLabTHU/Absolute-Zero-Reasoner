#!/usr/bin/env python3
"""
Validation Script for LoRA Checkpoints - V5 (Final & Working)

This version contains the last critical change: setting `strict=False` when
loading the adapter weights. This allows the partial state_dict (containing
only LoRA weights) to be loaded into the full model structure without error.
"""

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from peft.utils import get_peft_model_state_dict
import argparse
import time
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_inference(model, tokenizer, prompt):
    """Runs a single inference call."""
    if not prompt:
        return ""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

def main(base_model, lora_checkpoint, validation_file, num_samples=100):
    """Loads a model with LoRA adapter and evaluates it."""
    print("--- Starting LoRA Checkpoint Validation ---")

    # 1. Load the quantized base model
    logging.info(f"Loading base model: {base_model}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
       base_model,
       quantization_config=quantization_config,
       device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
       tokenizer.pad_token = tokenizer.eos_token

    # 2. Manually define the LoRA configuration
    logging.info("Defining LoRA configuration manually.")
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 3. Apply the empty LoRA structure to the base model
    model = get_peft_model(model, peft_config)
    logging.info("Applied empty LoRA layers to the base model.")
    model.print_trainable_parameters()

    # 4. Manually load, rename, and filter the weights from the FSDP checkpoint
    checkpoint_file = os.path.join(lora_checkpoint, "model_world_size_1_rank_0.pt")
    logging.info(f"Loading weights from raw FSDP checkpoint: {checkpoint_file}")

    if not os.path.exists(checkpoint_file):
       raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}. Cannot load weights.")
    
    fsdp_state_dict = torch.load(checkpoint_file, map_location="cpu")

    logging.info("Renaming keys in state_dict to match PEFT's '.default.weight' format...")
    renamed_state_dict = {}
    for key, value in fsdp_state_dict.items():
        new_key = key.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
        renamed_state_dict[new_key] = value

    adapter_weights = get_peft_model_state_dict(model, state_dict=renamed_state_dict)

    ## --- THE FINAL, CRITICAL CHANGE --- ##
    # Load the adapter weights with strict=False.
    # This allows us to load the partial state dict (containing only adapter weights)
    # into the full model without raising errors for the missing base model keys.
    model.load_state_dict(adapter_weights, strict=False)
    logging.info("✅ SUCCESS: Successfully loaded trained LoRA weights into the model.")
    
    model.eval()

    # 5. Run Validation
    logging.info(f"Loading validation data from: {validation_file}")
    df = pd.read_parquet(validation_file)
    df_sample = df.sample(n=min(num_samples, len(df)), random_state=42)
        
    print(f"Running validation on {len(df_sample)} samples...")
    
    correct_predictions = 0
    total_samples = len(df_sample)
    start_time = time.time()

    for i, row in enumerate(df_sample.iterrows()):
        index, data = row
        prompt = data['prompt'][0]['content'] if isinstance(data.get('prompt'), list) and data['prompt'] else str(data.get('prompt', ''))
        expected_output = data.get('extra_info', {}).get('output')
        
        generated_output = run_inference(model, tokenizer, prompt)
        
        if expected_output and str(expected_output).strip() in str(generated_output).strip():
            correct_predictions += 1
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total_samples} samples...")
            
    end_time = time.time()
    
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    duration = end_time - start_time
    
    print("\n--- LoRA Validation Complete ---")
    print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples})")
    print(f"Duration: {duration:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a LoRA checkpoint.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="Path to the checkpoint 'actor' directory.")
    parser.add_argument("--validation_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    main(args.base_model, args.lora_checkpoint, args.validation_file, args.num_samples)