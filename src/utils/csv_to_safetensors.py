"""
Utility to convert CSV vectors to safetensors format compatible with the centering system.

This script converts a CSV file containing a single vector (one value per line) 
to the same safetensors format used by the corpus mean system.
"""

import os
import sys
import time
from typing import Optional
import torch
from safetensors.torch import save_file


def csv_to_safetensors(
    csv_path: str,
    output_path: Optional[str] = None,
    tensor_name: str = "vector",
    d_model: int = 5120,
    max_tokens: Optional[int] = None,
) -> str:
    """Convert a CSV file to safetensors format compatible with centering system.
    
    Args:
        csv_path: Path to CSV file (one value per line, expecting d_model values)
        output_path: Output safetensors path (auto-generated if None)
        tensor_name: Name for the tensor in the safetensors file
        d_model: Expected model dimension (default 5120 for Pythia-12B)
        max_tokens: If provided, reshape to [d_model, max_tokens], otherwise [d_model]
        
    Returns:
        Path to the created safetensors file
        
    Raises:
        ValueError: If CSV doesn't have expected number of values
        FileNotFoundError: If CSV file doesn't exist
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV file (assuming one value per line, no header)
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
            lines = f.readlines()
        
        # Try to parse as simple one-value-per-line format first
        values = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove any remaining BOM characters that might have slipped through
            line = line.lstrip('\ufeff')
            
            # Check if line contains comma (multi-column CSV)
            if ',' in line:
                # Split by comma and take all values
                row_values = []
                for val in line.split(','):
                    val = val.strip().lstrip('\ufeff')  # Clean each value
                    if val:
                        row_values.append(float(val))
                values.extend(row_values)
            else:
                # Single value per line
                values.append(float(line))
                
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file {csv_path}: {e}")
    
    print(f"📊 Loaded {len(values)} values from {csv_path}")
    
    # Convert to tensor
    tensor = torch.tensor(values, dtype=torch.float32)
    
    # Reshape based on requirements
    if max_tokens is not None:
        # Reshape to [d_model, max_tokens] format (like corpus_mean)
        expected_size = d_model * max_tokens
        if len(values) != expected_size:
            raise ValueError(
                f"CSV has {len(values)} values, but expected {expected_size} "
                f"for shape [{d_model}, {max_tokens}]"
            )
        tensor = tensor.reshape(d_model, max_tokens)
        print(f"📐 Reshaped to [{d_model}, {max_tokens}] (corpus_mean format)")
    else:
        # Keep as 1D vector
        if len(values) != d_model:
            print(f"⚠️  Warning: CSV has {len(values)} values, expected {d_model} for d_model")
            print(f"    Proceeding with actual size: {len(values)}")
        print(f"📐 Keeping as 1D vector: [{len(values)}]")
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        timestamp = int(time.time())
        output_path = f"outputs/{base_name}_{timestamp}.safetensors"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to safetensors
    tensors_dict = {tensor_name: tensor}
    
    # If this is meant to be a corpus_mean replacement, add counts tensor
    if tensor_name == "corpus_mean" and max_tokens is not None:
        # Create a counts tensor (all ones, indicating each position has 1 sample)
        counts = torch.ones(max_tokens, dtype=torch.int64)
        tensors_dict["counts"] = counts
        print(f"📊 Added counts tensor: [{max_tokens}] (all ones)")
    
    save_file(tensors_dict, output_path)
    
    print(f"✅ Saved safetensors to: {output_path}")
    print(f"   Tensor '{tensor_name}': {tensor.shape}, dtype={tensor.dtype}")
    
    return output_path


def main():
    """Command line interface for CSV to safetensors conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert CSV vector to safetensors format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single vector CSV to safetensors
  python src/utils/csv_to_safetensors.py outputs/carrot_cake.csv
  
  # Convert to corpus_mean format (reshape to [5120, tokens])
  python src/utils/csv_to_safetensors.py outputs/carrot_cake.csv --corpus-mean --max-tokens 1
  
  # Custom output path and tensor name
  python src/utils/csv_to_safetensors.py outputs/carrot_cake.csv --output my_vector.safetensors --name my_vector
        """
    )
    
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("--output", "-o", help="Output safetensors path (auto-generated if not provided)")
    parser.add_argument("--name", "-n", default="vector", help="Tensor name in safetensors file (default: vector)")
    parser.add_argument("--d-model", type=int, default=5120, help="Model dimension (default: 5120)")
    parser.add_argument("--max-tokens", type=int, help="Reshape to [d_model, max_tokens] format")
    parser.add_argument("--corpus-mean", action="store_true", 
                       help="Format as corpus_mean (sets name='corpus_mean' and adds counts tensor)")
    
    args = parser.parse_args()
    
    # Handle corpus_mean flag
    tensor_name = args.name
    max_tokens = args.max_tokens
    
    if args.corpus_mean:
        tensor_name = "corpus_mean"
        if max_tokens is None:
            # Try to infer from CSV size
            try:
                with open(args.csv_path, 'r') as f:
                    num_lines = sum(1 for _ in f)
                if num_lines % args.d_model == 0:
                    max_tokens = num_lines // args.d_model
                    print(f"🔍 Inferred max_tokens={max_tokens} from CSV size {num_lines} and d_model={args.d_model}")
                else:
                    max_tokens = 1  # Assume single token
                    print(f"🔍 Using max_tokens=1 (CSV size {num_lines} doesn't divide evenly by d_model={args.d_model})")
            except Exception:
                max_tokens = 1
    
    try:
        output_path = csv_to_safetensors(
            csv_path=args.csv_path,
            output_path=args.output,
            tensor_name=tensor_name,
            d_model=args.d_model,
            max_tokens=max_tokens,
        )
        
        print(f"\n🎉 Conversion complete!")
        print(f"   Input:  {args.csv_path}")
        print(f"   Output: {output_path}")
        
        # Show how to use it
        if tensor_name == "corpus_mean":
            print(f"\n💡 To use as centering vector:")
            print(f"   from src.utils.centering import load_corpus_mean")
            print(f"   corpus_mean = load_corpus_mean('{output_path}')")
        else:
            print(f"\n💡 To load the tensor:")
            print(f"   from safetensors.torch import load_file")
            print(f"   tensors = load_file('{output_path}')")
            print(f"   my_vector = tensors['{tensor_name}']")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
