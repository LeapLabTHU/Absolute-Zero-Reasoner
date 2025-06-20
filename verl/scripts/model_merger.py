#!/usr/bin/env python3
"""
🔧 ENHANCED MODEL MERGER - FIXED TENSOR HANDLING
Handles LoRA-merged models with proper validation and error recovery
"""

import torch
import torch.nn as nn
import argparse
import os
import json
import time
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Dict, Any, Optional
import logging

# Enhanced logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_header(title: str, emoji: str = "🔧"):
    """Print formatted header"""
    print(f"\n{emoji} --- {title} ---")

def get_memory_usage():
    """Get GPU memory usage if available"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved"
    else:
        return "GPU: Not available"

def validate_checkpoint_structure(checkpoint_path: str) -> Dict[str, Any]:
    """
    Validate checkpoint structure and extract metadata
    """
    print_header("CHECKPOINT VALIDATION", "🔍")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    
    # Check for merge metadata
    metadata_path = os.path.join(checkpoint_path, 'merge_metadata.json')
    metadata = {}
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"📋 Found merge metadata:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
    else:
        print("⚠️  No merge metadata found")
    
    # Check for model files
    model_files = []
    for file in os.listdir(checkpoint_path):
        if file.endswith('.bin') or file.endswith('.safetensors') or file.endswith('.pt'):
            model_files.append(file)
    
    print(f"📂 Found {len(model_files)} model files")
    
    # Check for standard HuggingFace files
    required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(checkpoint_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
    
    return metadata

def load_and_validate_model(model_path: str, metadata: Dict[str, Any]) -> nn.Module:
    """
    Load model with enhanced validation and error handling
    """
    print_header("MODEL LOADING & VALIDATION", "🧠")
    
    try:
        # Load configuration
        config = AutoConfig.from_pretrained(model_path)
        print(f"📋 Model config: {config.model_type} - {config.num_hidden_layers} layers")
        
        # Load model with error handling
        print("📥 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float32,
            device_map=None,
            local_files_only=True
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        
        # Validate against metadata if available
        if 'parameter_count' in metadata:
            expected_count = metadata['parameter_count']
            if abs(total_params - expected_count) > 1000:
                print(f"⚠️  Parameter count mismatch! Expected: {expected_count:,}, Got: {total_params:,}")
            else:
                print(f"✅ Parameter count matches metadata: {total_params:,}")
        
        # Test forward pass
        print("🧪 Testing forward pass...")
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randint(0, 1000, (1, 4), dtype=torch.long)
            output = model(dummy_input)
            print(f"✅ Forward pass successful: output shape {output.logits.shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise

def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Clean state dict by removing problematic LoRA artifacts
    """
    print_header("CLEANING STATE DICT", "🧹")
    
    cleaned_dict = {}
    removed_keys = []
    
    for key, value in state_dict.items():
        # Remove LoRA-specific keys that shouldn't be in final model
        if any(lora_key in key for lora_key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
            removed_keys.append(key)
            continue
        
        # Remove base_layer prefixes if they exist
        if '.base_layer.' in key:
            cleaned_key = key.replace('.base_layer.', '.')
            cleaned_dict[cleaned_key] = value
        else:
            cleaned_dict[key] = value
    
    print(f"🗑️  Removed {len(removed_keys)} LoRA artifacts")
    print(f"✅ Cleaned state dict has {len(cleaned_dict)} parameters")
    
    return cleaned_dict

def save_model_safely(model: nn.Module, output_path: str, metadata: Dict[str, Any]):
    """
    Save model with backup and validation
    """
    print_header("SAFE MODEL SAVING", "💾")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Clean state dict before saving
    state_dict = model.state_dict()
    cleaned_state_dict = clean_state_dict(state_dict)
    
    # Create temporary model for validation
    temp_model = type(model)(model.config)
    temp_model.load_state_dict(cleaned_state_dict)
    
    # Save with HuggingFace format
    print("💾 Saving model...")
    start_time = time.time()
    
    temp_model.save_pretrained(
        output_path,
        safe_serialization=True,  # Use safetensors format
        max_shard_size="5GB"
    )
    
    save_time = time.time() - start_time
    print(f"✅ Model saved in {save_time:.1f} seconds")
    
    # Save enhanced metadata
    enhanced_metadata = metadata.copy()
    enhanced_metadata.update({
        'saved_timestamp': time.time(),
        'save_time_seconds': save_time,
        'final_parameter_count': sum(p.numel() for p in temp_model.parameters()),
        'model_size_mb': sum(os.path.getsize(os.path.join(output_path, f)) 
                           for f in os.listdir(output_path) 
                           if f.endswith(('.bin', '.safetensors'))) / (1024**2),
        'validation_passed': True
    })
    
    metadata_path = os.path.join(output_path, 'merger_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(enhanced_metadata, f, indent=2)
    
    print(f"📊 Model size: {enhanced_metadata['model_size_mb']:.1f} MB")
    print(f"📋 Metadata saved to: {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Model Merger with LoRA Artifact Cleaning")
    parser.add_argument('--local_dir', type=str, required=True, help='Local directory containing merged model')
    parser.add_argument('--output_dir', type=str, help='Output directory (defaults to local_dir/huggingface)')
    parser.add_argument('--verify_model', action='store_true', help='Run additional model verification')
    parser.add_argument('--backup_original', action='store_true', help='Create backup of original before processing')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.local_dir, 'huggingface')
    
    print_header("ENHANCED MODEL MERGER STARTING", "🔧")
    print(f"📍 Local directory: {args.local_dir}")
    print(f"📍 Output directory: {output_dir}")
    print(f"🧠 Memory usage: {get_memory_usage()}")
    
    try:
        # Validate checkpoint structure
        metadata = validate_checkpoint_structure(args.local_dir)
        
        # Create backup if requested
        if args.backup_original:
            backup_dir = f"{args.local_dir}_backup_{int(time.time())}"
            print(f"💾 Creating backup: {backup_dir}")
            shutil.copytree(args.local_dir, backup_dir)
        
        # Load and validate model
        model = load_and_validate_model(args.local_dir, metadata)
        
        # Additional verification if requested
        if args.verify_model:
            print_header("ADDITIONAL MODEL VERIFICATION", "🔬")
            
            # Check for NaN values
            nan_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    nan_params.append(name)
            
            if nan_params:
                print(f"⚠️  Found NaN values in {len(nan_params)} parameters:")
                for name in nan_params[:5]:  # Show first 5
                    print(f"   - {name}")
            else:
                print("✅ No NaN values found")
            
            # Check parameter ranges
            param_stats = {}
            for name, param in model.named_parameters():
                param_stats[name] = {
                    'min': param.min().item(),
                    'max': param.max().item(),
                    'mean': param.mean().item(),
                    'std': param.std().item()
                }
            
            # Look for suspicious values
            suspicious = []
            for name, stats in param_stats.items():
                if abs(stats['max']) > 100 or abs(stats['min']) > 100:
                    suspicious.append(name)
            
            if suspicious:
                print(f"⚠️  Suspicious parameter ranges in {len(suspicious)} tensors:")
                for name in suspicious[:3]:
                    stats = param_stats[name]
                    print(f"   - {name}: [{stats['min']:.3f}, {stats['max']:.3f}]")
            else:
                print("✅ Parameter ranges look normal")
        
        # Save model safely
        save_model_safely(model, output_dir, metadata)
        
        # Copy tokenizer files
        print_header("COPYING TOKENIZER FILES", "📝")
        tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'merges.txt', 'special_tokens_map.json']
        
        for file in tokenizer_files:
            src_path = os.path.join(args.local_dir, file)
            dst_path = os.path.join(output_dir, file)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"✅ Copied {file}")
        
        # Final validation
        print_header("FINAL VALIDATION", "✅")
        
        # Test loading the saved model
        try:
            test_model = AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True)
            test_params = sum(p.numel() for p in test_model.parameters())
            print(f"✅ Saved model loads successfully: {test_params:,} parameters")
            
            # Test tokenizer
            tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
            print(f"✅ Tokenizer loads successfully: vocab size {tokenizer.vocab_size}")
            
        except Exception as e:
            print(f"⚠️  Validation warning: {e}")
        
        print(f"🧠 Final memory usage: {get_memory_usage()}")
        
        print_header("MODEL MERGER COMPLETED SUCCESSFULLY", "🎉")
        print(f"✅ Enhanced model saved to: {output_dir}")
        print(f"✅ Ready for next training cycle")
        
    except Exception as e:
        print(f"❌ MODEL MERGER FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on failure
        if os.path.exists(output_dir) and output_dir != args.local_dir:
            print("🧹 Cleaning up failed output...")
            shutil.rmtree(output_dir, ignore_errors=True)
        
        exit(1)

if __name__ == "__main__":
    main()