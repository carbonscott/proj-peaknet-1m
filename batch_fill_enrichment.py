#!/usr/bin/env python3
"""
Batch Enrichment Filler

Processes all classification results to create fully enriched documents.
Takes classification JSON files and corresponding enrichment markdown files,
then creates fully enriched documents with all __ANSWER__ placeholders filled.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import re


def extract_experiment_id(classification_filename: str) -> str:
    """Extract experiment ID from classification filename"""
    # Remove '_classifications.json' suffix
    if classification_filename.endswith('_classifications.json'):
        return classification_filename[:-21]
    else:
        # Fallback: remove .json extension
        return Path(classification_filename).stem


def main():
    parser = argparse.ArgumentParser(
        description="Process classification results to create fully enriched documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default directories
  python batch_fill_enrichment.py

  # Custom directories
  python batch_fill_enrichment.py --classification-dir results/ --output-dir enriched/

  # Process limited files with force overwrite
  python batch_fill_enrichment.py --max-files 10 --force

  # Full custom setup
  python batch_fill_enrichment.py --classification-dir my_results/ --enrichment-dir my_enriched/ --output-dir final_docs/ --force
        """
    )

    # Directory arguments
    parser.add_argument('--classification-dir', default='batch_crystallography_results',
                       help='Directory containing classification JSON files (default: batch_crystallography_results)')
    parser.add_argument('--enrichment-dir', default='processed_experiments', 
                       help='Directory containing enrichment markdown files (default: processed_experiments)')
    parser.add_argument('--output-dir', default='fully_enriched_experiments',
                       help='Directory to save fully enriched files (default: fully_enriched_experiments)')

    # Optional arguments
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing output files')
    parser.add_argument('--max-files', type=int, 
                       help='Maximum number of files to process (for testing)')

    args = parser.parse_args()

    # Configuration from arguments
    classification_dir = args.classification_dir
    enrichment_dir = args.enrichment_dir
    output_dir = args.output_dir

    print("📝 LCLS Batch Enrichment Filler")
    print("=" * 50)

    # Get all classification files
    classification_path = Path(classification_dir)
    if not classification_path.exists():
        print(f"❌ Error: Classification directory not found: {classification_dir}")
        sys.exit(1)

    classification_files = list(classification_path.glob("*_classifications.json"))
    print(f"✓ Found {len(classification_files)} classification files")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"✓ Output directory: {output_dir}/")
    print()

    # Limit files if max_files specified
    if args.max_files:
        classification_files = classification_files[:args.max_files]
        print(f"✓ Limited to first {len(classification_files)} files (--max-files {args.max_files})")

    # Process each classification file
    processed_count = 0
    skipped_count = 0

    for i, classification_file in enumerate(classification_files, 1):
        # Extract experiment ID
        exp_id = extract_experiment_id(classification_file.name)
        print(f"[{i:2d}/{len(classification_files)}] Processing: {exp_id}")

        # Look for corresponding enrichment file
        enrichment_file = Path(enrichment_dir) / f"{exp_id}_enrichment.md"
        output_file = output_path / f"{exp_id}_full_enrichment.md"

        if not enrichment_file.exists():
            print(f"  ⚠️  Skipped - enrichment file not found: {enrichment_file}")
            skipped_count += 1
            continue

        # Check if output already exists (skip unless --force)
        if output_file.exists() and not args.force:
            print(f"  ⚠️  Skipped - output already exists: {output_file} (use --force to overwrite)")
            skipped_count += 1
            continue

        # Run fill_enrichment.py
        try:
            cmd = [
                "python", "fill_enrichment.py",
                str(enrichment_file),
                str(classification_file),
                "-o", str(output_file)
            ]

            print(f"  🚀 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print(f"  ✅ Completed: {output_file}")
                processed_count += 1
            else:
                print(f"  ❌ Failed with return code {result.returncode}")
                if result.stderr:
                    # Extract relevant error message
                    error_lines = result.stderr.strip().split('\n')
                    relevant_error = [line for line in error_lines if '❌' in line or 'Error:' in line]
                    if relevant_error:
                        print(f"     Error: {relevant_error[-1]}")
                    else:
                        print(f"     Error: {error_lines[-1] if error_lines else 'Unknown error'}")
                skipped_count += 1

        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout after 60 seconds - skipping {exp_id}")
            skipped_count += 1
        except Exception as e:
            print(f"  ❌ Error running fill_enrichment.py: {e}")
            skipped_count += 1

        print()

    # Summary
    print("=" * 50)
    print("📊 BATCH ENRICHMENT FILLING COMPLETE")
    print(f"✅ Successfully processed: {processed_count}")
    print(f"⚠️  Skipped: {skipped_count}")
    print(f"📁 Classification files found: {len(classification_files)}")
    print(f"📁 Fully enriched files created: {processed_count}")

    if processed_count > 0:
        print(f"\n🔍 To view results:")
        print(f"   ls {output_dir}/*.md")
        print(f"   # Each file contains complete run classifications")

        # Show a few examples
        output_files = list(output_path.glob("*_full_enrichment.md"))
        if output_files:
            print(f"\n📋 Sample outputs:")
            for output_file in sorted(output_files)[:5]:
                print(f"   ✓ {output_file.name}")
            if len(output_files) > 5:
                print(f"   ... and {len(output_files) - 5} more")

    if skipped_count > 0:
        print(f"\n⚠️  Note: {skipped_count} files were skipped")
        print("   - Missing enrichment files")
        print("   - Already processed files")
        print("   - Processing errors")


if __name__ == "__main__":
    main()
