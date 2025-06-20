#!/usr/bin/env python3
"""
AZR Parquet Data Merger
Combines newly generated training data with existing datasets for AZR training.

Usage:
python merge_parquet_data.py --checkpoint-dir checkpoints/code_io/azr/0.5b_coder_seed_generation/test_answer/Qwen2.5-Coder-0.5B/answer_conditional/code/ --output-dir data/code_reason/
"""

import pandas as pd
import argparse
import os
import logging
from pathlib import Path
from typing import List, Dict
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParquetDataMerger:
    def __init__(self, checkpoint_dir: str, output_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the expected file patterns
        self.file_patterns = [
            'train_gen_code_f.parquet',
            'train_gen_code_i.parquet', 
            'train_gen_code_o.parquet',
            'train_pred_code_f.parquet',
            'train_pred_code_i.parquet',
            'train_pred_code_o.parquet'
        ]
    
    def find_checkpoint_files(self) -> Dict[str, Path]:
        """Find all generated parquet files in checkpoint directory."""
        found_files = {}
        
        logger.info(f"Scanning checkpoint directory: {self.checkpoint_dir}")
        
        for pattern in self.file_patterns:
            checkpoint_file = self.checkpoint_dir / pattern
            if checkpoint_file.exists():
                found_files[pattern] = checkpoint_file
                logger.info(f"Found: {pattern} ({self.get_file_size(checkpoint_file)})")
            else:
                logger.warning(f"Missing: {pattern}")
        
        return found_files
    
    def get_file_size(self, file_path: Path) -> str:
        """Get human-readable file size."""
        size = file_path.stat().st_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def load_dataframe(self, file_path: Path, description: str) -> pd.DataFrame:
        """Load parquet file with error handling."""
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {description}: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load {description} from {file_path}: {e}")
            return pd.DataFrame()
    
    def merge_single_file_type(self, pattern: str, checkpoint_file: Path) -> bool:
        """Merge a specific file type (e.g., train_gen_code_f.parquet)."""
        logger.info(f"\n=== Processing {pattern} ===")
        
        # Load new data from checkpoint
        new_df = self.load_dataframe(checkpoint_file, f"new {pattern}")
        if new_df.empty:
            logger.warning(f"No data in new {pattern}, skipping")
            return False
        
        # Check for existing data
        existing_file = self.output_dir / "test_answer.parquet"
        base_name = pattern.replace('.parquet', '')
        
        if existing_file.exists():
            logger.info(f"Found existing training data: {existing_file}")
            existing_df = self.load_dataframe(existing_file, "existing training data")
            
            if not existing_df.empty:
                # Filter existing data to match current file type if needed
                # For now, we'll merge all data together
                logger.info("Merging with existing data...")
                
                # Combine dataframes
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # Remove duplicates if any (based on all columns)
                original_len = len(combined_df)
                combined_df = combined_df.drop_duplicates()
                dedupe_removed = original_len - len(combined_df)
                
                if dedupe_removed > 0:
                    logger.info(f"Removed {dedupe_removed} duplicate rows")
                
                logger.info(f"Final dataset: {len(combined_df)} rows")
            else:
                combined_df = new_df
        else:
            logger.info("No existing training data found, using new data only")
            combined_df = new_df
        
        # Save merged data
        output_file = self.output_dir / "test_answer.parquet"
        combined_df.to_parquet(output_file, index=False)
        logger.info(f"Saved merged data to: {output_file}")
        
        return True
    
    def create_individual_files(self, checkpoint_files: Dict[str, Path]):
        """Create individual merged files for each data type."""
        logger.info("\n=== Creating individual merged files ===")
        
        for pattern, checkpoint_file in checkpoint_files.items():
            output_file = self.output_dir / pattern
            
            # Load and save new data
            new_df = self.load_dataframe(checkpoint_file, f"new {pattern}")
            if not new_df.empty:
                new_df.to_parquet(output_file, index=False)
                logger.info(f"Created: {output_file} ({len(new_df)} rows)")
    
    def merge_all_data(self) -> bool:
        """Main method to merge all parquet data."""
        logger.info("=== AZR Parquet Data Merger ===")
        
        # Find checkpoint files
        checkpoint_files = self.find_checkpoint_files()
        
        if not checkpoint_files:
            logger.error("No checkpoint files found!")
            return False
        
        logger.info(f"Found {len(checkpoint_files)} checkpoint files")
        
        # Strategy 1: Merge all new data into single test_answer.parquet
        logger.info("\n=== Merging all data into test_answer.parquet ===")
        
        all_new_data = []
        total_rows = 0
        
        for pattern, checkpoint_file in checkpoint_files.items():
            df = self.load_dataframe(checkpoint_file, pattern)
            if not df.empty:
                all_new_data.append(df)
                total_rows += len(df)
        
        if all_new_data:
            # Combine all new data
            combined_new = pd.concat(all_new_data, ignore_index=True)
            logger.info(f"Combined new data: {len(combined_new)} rows")
            
            # Load existing data if available
            existing_file = self.output_dir / "test_answer.parquet"
            if existing_file.exists():
                existing_df = self.load_dataframe(existing_file, "existing test_answer.parquet")
                if not existing_df.empty:
                    # Merge with existing
                    final_df = pd.concat([existing_df, combined_new], ignore_index=True)
                    
                    # Remove duplicates
                    original_len = len(final_df)
                    final_df = final_df.drop_duplicates()
                    dedupe_removed = original_len - len(final_df)
                    
                    if dedupe_removed > 0:
                        logger.info(f"Removed {dedupe_removed} duplicate rows")
                else:
                    final_df = combined_new
            else:
                final_df = combined_new
            
            # Save final merged data
            final_df.to_parquet(existing_file, index=False)
            logger.info(f"SUCCESS: Saved merged training data to {existing_file}")
            logger.info(f"Final dataset: {len(final_df)} rows, {len(final_df.columns)} columns")
            
            # Also create individual files for reference
            self.create_individual_files(checkpoint_files)
            
            return True
        
        return False
    
    def create_backup(self):
        """Create backup of existing data before merging."""
        existing_file = self.output_dir / "test_answer.parquet"
        if existing_file.exists():
            backup_file = self.output_dir / f"test_answer_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            import shutil
            shutil.copy2(existing_file, backup_file)
            logger.info(f"Created backup: {backup_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge AZR generated parquet data with existing training data")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing generated parquet files")
    parser.add_argument("--output-dir", required=True, help="Directory to save merged training data")
    parser.add_argument("--backup", action="store_true", help="Create backup of existing data")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be merged without actually doing it")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_dir):
        logger.error(f"Checkpoint directory does not exist: {args.checkpoint_dir}")
        sys.exit(1)
    
    merger = ParquetDataMerger(args.checkpoint_dir, args.output_dir)
    
    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        checkpoint_files = merger.find_checkpoint_files()
        logger.info(f"Would process {len(checkpoint_files)} files")
        for pattern, file_path in checkpoint_files.items():
            df = merger.load_dataframe(file_path, pattern)
            logger.info(f"Would merge {len(df)} rows from {pattern}")
        return
    
    if args.backup:
        merger.create_backup()
    
    success = merger.merge_all_data()
    
    if success:
        logger.info("\n=== MERGE COMPLETED SUCCESSFULLY ===")
        logger.info("You can now run AZR training with the updated dataset!")
    else:
        logger.error("\n=== MERGE FAILED ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
