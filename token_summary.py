#!/usr/bin/env python3
"""
Token Usage Summary Script

Aggregates token usage statistics from multiple classification JSON files.
Provides detailed breakdown and total usage across all experiments.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple


def extract_token_data(file_path: Path) -> Tuple[str, Dict[str, int], bool]:
    """
    Extract token usage data from a classification JSON file
    
    Args:
        file_path: Path to classification JSON file
        
    Returns:
        Tuple of (experiment_id, token_data, success_flag)
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        experiment_id = data.get('experiment_id', file_path.stem)
        token_usage = data.get('processing_info', {}).get('token_usage', {})
        
        # Extract token counts with defaults
        token_data = {
            'total_tokens': token_usage.get('total_tokens', 0),
            'prompt_tokens': token_usage.get('prompt_tokens', 0),
            'completion_tokens': token_usage.get('completion_tokens', 0),
            'chunks': len(token_usage.get('per_chunk_tokens', [])),
            'total_runs': data.get('total_runs', 0)
        }
        
        return experiment_id, token_data, True
        
    except FileNotFoundError:
        return file_path.name, {}, False
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON error in {file_path.name}: {e}", file=sys.stderr)
        return file_path.name, {}, False
    except Exception as e:
        print(f"⚠️  Error processing {file_path.name}: {e}", file=sys.stderr)
        return file_path.name, {}, False


def calculate_cost_estimate(total_tokens: int, prompt_tokens: int, completion_tokens: int) -> Dict[str, float]:
    """
    Calculate rough cost estimates based on typical API pricing
    
    Note: This is an approximation - actual costs may vary
    """
    # Rough estimates based on Claude API pricing (example rates)
    prompt_cost_per_1k = 0.015    # $0.015 per 1K prompt tokens
    completion_cost_per_1k = 0.075 # $0.075 per 1K completion tokens
    
    prompt_cost = (prompt_tokens / 1000) * prompt_cost_per_1k
    completion_cost = (completion_tokens / 1000) * completion_cost_per_1k
    total_cost = prompt_cost + completion_cost
    
    return {
        'prompt_cost': prompt_cost,
        'completion_cost': completion_cost,
        'total_cost': total_cost
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate token usage statistics from classification JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all files in batch results
  python token_summary.py batch_crystallography_results/*.json
  
  # Analyze specific files
  python token_summary.py file1.json file2.json file3.json
  
  # Show detailed per-file breakdown
  python token_summary.py --detailed batch_crystallography_results/*.json
  
  # Include cost estimates
  python token_summary.py --cost batch_crystallography_results/*.json
        """
    )
    
    # Required arguments
    parser.add_argument('files', nargs='+', type=Path,
                       help='Classification JSON files to analyze')
    
    # Optional arguments
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed per-file breakdown')
    parser.add_argument('--cost', action='store_true',
                       help='Include rough cost estimates')
    
    args = parser.parse_args()
    
    print("📊 Token Usage Summary")
    print("=" * 50)
    print(f"Processing {len(args.files)} files...")
    print()
    
    # Process all files
    successful_files = []
    failed_files = []
    total_stats = {
        'total_tokens': 0,
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_chunks': 0,
        'total_runs': 0
    }
    
    for file_path in args.files:
        experiment_id, token_data, success = extract_token_data(file_path)
        
        if success and token_data.get('total_tokens', 0) > 0:
            successful_files.append((experiment_id, file_path.name, token_data))
            total_stats['total_tokens'] += token_data['total_tokens']
            total_stats['prompt_tokens'] += token_data['prompt_tokens']
            total_stats['completion_tokens'] += token_data['completion_tokens']
            total_stats['total_chunks'] += token_data['chunks']
            total_stats['total_runs'] += token_data['total_runs']
        else:
            failed_files.append(file_path.name)
    
    # Show detailed breakdown if requested
    if args.detailed and successful_files:
        print("📋 Per-file breakdown:")
        print("-" * 50)
        for experiment_id, filename, token_data in sorted(successful_files):
            print(f"✓ {experiment_id:15} | {token_data['total_tokens']:6,} tokens "
                  f"({token_data['prompt_tokens']:5,} prompt + {token_data['completion_tokens']:4,} completion) "
                  f"| {token_data['total_runs']:3} runs | {token_data['chunks']:2} chunks")
        print()
    
    # Show summary statistics
    print("🎯 Summary Statistics:")
    print("-" * 50)
    print(f"📁 Files processed:     {len(successful_files):,}/{len(args.files):,}")
    print(f"🔬 Total experiments:   {len(successful_files):,}")
    print(f"🏃 Total runs:          {total_stats['total_runs']:,}")
    print(f"📦 Total chunks:        {total_stats['total_chunks']:,}")
    print()
    
    # Token usage summary
    print("💎 Token Usage:")
    print("-" * 50)
    print(f"📊 Total tokens:        {total_stats['total_tokens']:,}")
    print(f"📝 Prompt tokens:       {total_stats['prompt_tokens']:,}")
    print(f"💬 Completion tokens:   {total_stats['completion_tokens']:,}")
    
    # Calculate averages
    if successful_files:
        avg_total = total_stats['total_tokens'] // len(successful_files)
        avg_prompt = total_stats['prompt_tokens'] // len(successful_files)
        avg_completion = total_stats['completion_tokens'] // len(successful_files)
        
        print(f"📈 Average per file:    {avg_total:,} tokens ({avg_prompt:,} prompt + {avg_completion:,} completion)")
        
        if total_stats['total_runs'] > 0:
            avg_tokens_per_run = total_stats['total_tokens'] // total_stats['total_runs']
            print(f"🎯 Average per run:     {avg_tokens_per_run:,} tokens")
    
    # Cost estimates if requested
    if args.cost and total_stats['total_tokens'] > 0:
        print()
        print("💰 Cost Estimates (Approximate):")
        print("-" * 50)
        costs = calculate_cost_estimate(
            total_stats['total_tokens'],
            total_stats['prompt_tokens'], 
            total_stats['completion_tokens']
        )
        print(f"💸 Prompt costs:        ${costs['prompt_cost']:.3f}")
        print(f"💸 Completion costs:    ${costs['completion_cost']:.3f}")
        print(f"💸 Total estimated:     ${costs['total_cost']:.3f}")
        print("   (Note: Actual costs may vary based on API provider and pricing)")
    
    # Show any failed files
    if failed_files:
        print()
        print("⚠️  Failed to process:")
        print("-" * 50)
        for filename in sorted(failed_files):
            print(f"❌ {filename}")
    
    print()
    print("=" * 50)
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()