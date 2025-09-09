#!/usr/bin/env python3
"""
Script to inspect training data downloaded from Modal volume.

Usage:
    python inspect_data.py [path_to_pkl_file]
    
If no path provided, looks for 'training_data.pkl' in current directory.
"""

import pickle
import sys
from pathlib import Path
from typing import List


def load_training_data(file_path: str = "training_data.pkl") -> List[str]:
    """Load training data from pickle file."""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ Loaded {len(data)} training samples from {file_path}")
        return data
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Make sure to download the data first with:")
        print("modal volume get pile-dataset-cache training_data.pkl ./training_data.pkl")
        return []
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return []


def inspect_data(data: List[str]):
    """Inspect and analyze the training data."""
    if not data:
        return
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(data)}")
    print(f"   Average length: {sum(len(text) for text in data) / len(data):.1f} chars")
    print(f"   Shortest sample: {min(len(text) for text in data)} chars")
    print(f"   Longest sample: {max(len(text) for text in data)} chars")
    
    # Word count statistics
    word_counts = [len(text.split()) for text in data]
    print(f"   Average words: {sum(word_counts) / len(word_counts):.1f}")
    print(f"   Shortest (words): {min(word_counts)}")
    print(f"   Longest (words): {max(word_counts)}")


def show_samples(data: List[str], num_samples: int = 5):
    """Show sample texts from the dataset."""
    if not data:
        return
    
    print(f"\n📝 Sample Texts (first {min(num_samples, len(data))}):")
    for i, text in enumerate(data[:num_samples]):
        print(f"\n--- Sample {i + 1} ---")
        print(f"Length: {len(text)} chars, {len(text.split())} words")
        print(f"Preview: {text[:300]}...")
        if len(text) > 300:
            print(f"[... {len(text) - 300} more characters]")


def search_text(data: List[str], query: str, max_results: int = 10):
    """Search for text containing a query string."""
    if not data:
        return
    
    matches = []
    for i, text in enumerate(data):
        if query.lower() in text.lower():
            matches.append((i, text))
            if len(matches) >= max_results:
                break
    
    print(f"\n🔍 Search Results for '{query}' (found {len(matches)} matches):")
    for i, (idx, text) in enumerate(matches):
        print(f"\n--- Match {i + 1} (Sample {idx}) ---")
        # Show context around the match
        text_lower = text.lower()
        pos = text_lower.find(query.lower())
        start = max(0, pos - 100)
        end = min(len(text), pos + len(query) + 100)
        context = text[start:end]
        print(f"...{context}...")


def main():
    # Get file path from command line or use default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "training_data.pkl"
    
    print(f"🔍 Inspecting training data: {file_path}")
    
    # Load and inspect data
    data = load_training_data(file_path)
    if not data:
        return
    
    # Show statistics
    inspect_data(data)
    
    # Show sample texts
    show_samples(data, num_samples=3)
    
    # Interactive mode
    print(f"\n🎯 Interactive Commands:")
    print("   'samples N' - Show N sample texts")
    print("   'search QUERY' - Search for text containing QUERY")
    print("   'stats' - Show statistics again") 
    print("   'quit' - Exit")
    
    while True:
        try:
            cmd = input(f"\n> ").strip()
            if not cmd:
                continue
            
            if cmd.lower() in ['quit', 'exit', 'q']:
                break
            elif cmd.lower() == 'stats':
                inspect_data(data)
            elif cmd.startswith('samples '):
                try:
                    n = int(cmd.split()[1])
                    show_samples(data, n)
                except (ValueError, IndexError):
                    print("Usage: samples N (where N is a number)")
            elif cmd.startswith('search '):
                query = ' '.join(cmd.split()[1:])
                if query:
                    search_text(data, query)
                else:
                    print("Usage: search QUERY")
            else:
                print("Unknown command. Try 'samples N', 'search QUERY', 'stats', or 'quit'")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()