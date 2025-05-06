#!/usr/bin/env python3
"""
Parquet schema inspector - helps understand what's in our parquet files
"""

import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from pathlib import Path

def inspect_parquet_file(file_path):
    """Inspect a single parquet file and print its schema and sample data"""
    print(f"\n{'='*80}")
    print(f"Inspecting file: {file_path}")
    print(f"{'='*80}")
    
    # Get file size
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    print(f"File size: {file_size:.2f} MB")
    
    # Read metadata only (without loading data)
    try:
        parquet_file = pq.ParquetFile(file_path)
        schema = parquet_file.schema
        
        # Print schema
        print("\nSchema:")
        for i, field in enumerate(schema):
            print(f"  {i+1}. {field.name}: {field.type}")
        
        # Get number of rows
        num_rows = parquet_file.metadata.num_rows
        print(f"\nNumber of rows: {num_rows}")
    except Exception as e:
        print(f"Error reading parquet metadata: {e}")
    
    # Try to read the actual data
    try:
        # Read with pandas
        print("\nReading with pandas...")
        df = pd.read_parquet(file_path)
        
        # Print DataFrame info
        print("\nDataFrame info:")
        print(f"  Shape: {df.shape}")
        print("  Columns:", df.columns.tolist())
        
        # Print a small sample of the data
        print("\nData sample (first 2 rows):")
        sample = df.head(2)
        
        # For each column, show data type and sample value
        for col in df.columns:
            sample_val = sample[col].iloc[0] if len(sample) > 0 else None
            # If sample value is very large, truncate it
            if isinstance(sample_val, (list, dict)) and len(str(sample_val)) > 100:
                sample_val = str(sample_val)[:100] + "..."
            print(f"  {col} ({df[col].dtype}): {sample_val}")
        
        # Check if there's a parent directory with subject info
        parent_dir = os.path.dirname(file_path)
        dir_name = os.path.basename(parent_dir)
        if dir_name.startswith("SubjectID="):
            subject_id = dir_name.split("=")[1]
            print(f"\nPartition information from directory name: SubjectID={subject_id}")
            
            # Check if SubjectID is also in the DataFrame
            if "SubjectID" in df.columns:
                print(f"SubjectID column exists in data: {df['SubjectID'].iloc[0]}")
            else:
                print("SubjectID is in directory name but NOT in DataFrame columns")
    
    except Exception as e:
        print(f"Error reading parquet data: {e}")

def inspect_directory(dir_path, pattern=None):
    """Inspect all parquet files in a directory"""
    path = Path(dir_path)
    
    if pattern:
        pattern_path = path / pattern
        parquet_files = glob.glob(str(pattern_path))
    else:
        pattern_path = path / "**/*.parquet"
        parquet_files = glob.glob(str(pattern_path), recursive=True)
    
    if not parquet_files:
        print(f"No parquet files found in {dir_path} with pattern {pattern}")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Sort files for consistent output
    parquet_files.sort()
    
    # Inspect only the first few files to avoid info overload
    max_files = 3
    for i, file_path in enumerate(parquet_files[:max_files]):
        inspect_parquet_file(file_path)
    
    if len(parquet_files) > max_files:
        print(f"\nShowing only {max_files} of {len(parquet_files)} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Parquet files to understand their schema")
    parser.add_argument("directory", help="Directory containing parquet files")
    parser.add_argument("--pattern", "-p", help="Glob pattern to filter files (relative to directory)")
    parser.add_argument("--file", "-f", help="Inspect a specific parquet file")
    
    args = parser.parse_args()
    
    if args.file:
        inspect_parquet_file(args.file)
    else:
        inspect_directory(args.directory, args.pattern)

