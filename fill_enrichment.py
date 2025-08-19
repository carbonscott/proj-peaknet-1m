#!/usr/bin/env python3
"""
Fill Enrichment Script

Takes an enrichment markdown file with __ANSWER__ placeholders and a classification 
JSON file, then creates a fully enriched markdown file with all placeholders filled.
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Dict, Optional


def load_classification_data(classification_file: Path) -> Dict[int, Dict[str, str]]:
    """
    Load classification data and create run_number mapping
    
    Args:
        classification_file: Path to classification JSON file
        
    Returns:
        Dictionary mapping run_number to classification data
    """
    try:
        with open(classification_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Classification file not found: {classification_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in classification file: {e}")
        sys.exit(1)
    
    # Build mapping from run_number to classification data
    run_mapping = {}
    classifications = data.get('classifications', [])
    
    for classification in classifications:
        run_num = classification.get('run_number')
        if run_num is not None:
            run_mapping[run_num] = {
                'classification': classification.get('classification', ''),
                'confidence': classification.get('confidence', ''),
                'key_evidence': classification.get('key_evidence', '')
            }
    
    print(f"✓ Loaded classifications for {len(run_mapping)} runs")
    return run_mapping


def process_enrichment_content(content: str, run_mapping: Dict[int, Dict[str, str]]) -> str:
    """
    Process enrichment content and fill __ANSWER__ placeholders
    
    Args:
        content: Original enrichment markdown content
        run_mapping: Mapping of run_number to classification data
        
    Returns:
        Processed content with placeholders filled
    """
    lines = content.split('\n')
    processed_lines = []
    current_run = None
    filled_count = 0
    
    for line in lines:
        # Check if this is a run header
        run_match = re.match(r'^### Run (\d+)', line)
        if run_match:
            current_run = int(run_match.group(1))
            processed_lines.append(line)
            continue
        
        # Process classification placeholders if we're in a run section
        if current_run is not None and current_run in run_mapping:
            run_data = run_mapping[current_run]
            
            # Replace classification placeholder
            if '**Run classification**: __ANSWER__' in line:
                line = line.replace('**Run classification**: __ANSWER__', 
                                  f"**Run classification**: {run_data['classification']}")
                filled_count += 1
            
            # Replace confidence placeholder
            elif '**Confidence**: __ANSWER__' in line:
                line = line.replace('**Confidence**: __ANSWER__', 
                                  f"**Confidence**: {run_data['confidence']}")
                filled_count += 1
            
            # Replace key evidence placeholder
            elif '**Key evidence**: __ANSWER__' in line:
                line = line.replace('**Key evidence**: __ANSWER__', 
                                  f"**Key evidence**: {run_data['key_evidence']}")
                filled_count += 1
        
        processed_lines.append(line)
    
    print(f"✓ Filled {filled_count} placeholders")
    return '\n'.join(processed_lines)


def generate_output_path(enrichment_file: Path) -> Path:
    """Generate output path in fully_enriched_experiments directory"""
    output_dir = Path("fully_enriched_experiments")
    
    # Extract experiment ID from filename
    filename = enrichment_file.name
    if filename.endswith('_enrichment.md'):
        exp_id = filename[:-15]  # Remove '_enrichment.md'
        output_filename = f"{exp_id}_full_enrichment.md"
    else:
        # Fallback if filename doesn't match expected pattern
        output_filename = f"{enrichment_file.stem}_full_enrichment.md"
    
    return output_dir / output_filename


def main():
    parser = argparse.ArgumentParser(
        description="Fill __ANSWER__ placeholders in enrichment files with classification results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python fill_enrichment.py processed_experiments/mfx101080524_enrichment.md batch_crystallography_results/mfx101080524_classifications.json
  
  # Specify custom output file
  python fill_enrichment.py enrichment.md classifications.json -o custom_output.md
  
  # Process single experiment
  python fill_enrichment.py processed_experiments/cxip1001524_enrichment.md batch_crystallography_results/cxip1001524_classifications.json
        """
    )
    
    # Required arguments
    parser.add_argument('enrichment_file', type=Path,
                       help='Input enrichment markdown file with __ANSWER__ placeholders')
    parser.add_argument('classification_file', type=Path,
                       help='Input classification JSON file with results')
    
    # Optional arguments
    parser.add_argument('-o', '--output', type=Path,
                       help='Output fully enriched markdown file (default: auto-generated in fully_enriched_experiments/)')
    
    args = parser.parse_args()
    
    print("📝 LCLS Enrichment Placeholder Filler")
    print("=" * 50)
    
    # Validate input files
    if not args.enrichment_file.exists():
        print(f"❌ Error: Enrichment file not found: {args.enrichment_file}")
        sys.exit(1)
    
    if not args.classification_file.exists():
        print(f"❌ Error: Classification file not found: {args.classification_file}")
        sys.exit(1)
    
    # Generate output path if not specified
    if not args.output:
        args.output = generate_output_path(args.enrichment_file)
    
    # Create output directory if it doesn't exist
    args.output.parent.mkdir(exist_ok=True)
    print(f"✓ Output directory: {args.output.parent}/")
    
    # Load classification data
    print(f"📖 Reading classification file: {args.classification_file}")
    run_mapping = load_classification_data(args.classification_file)
    
    # Read enrichment content
    print(f"📖 Reading enrichment file: {args.enrichment_file}")
    try:
        with open(args.enrichment_file, 'r') as f:
            enrichment_content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Could not read enrichment file")
        sys.exit(1)
    
    # Process content
    print("🔄 Processing placeholders...")
    filled_content = process_enrichment_content(enrichment_content, run_mapping)
    
    # Write output
    print(f"💾 Writing output: {args.output}")
    try:
        with open(args.output, 'w') as f:
            f.write(filled_content)
    except Exception as e:
        print(f"❌ Error writing output file: {e}")
        sys.exit(1)
    
    print("=" * 50)
    print("✅ ENRICHMENT FILLING COMPLETE")
    print(f"📁 Input enrichment: {args.enrichment_file}")
    print(f"📁 Input classifications: {args.classification_file}")
    print(f"📁 Output filled: {args.output}")
    
    # Basic stats
    original_placeholders = enrichment_content.count('__ANSWER__')
    remaining_placeholders = filled_content.count('__ANSWER__')
    filled_placeholders = original_placeholders - remaining_placeholders
    
    print(f"📊 Placeholders filled: {filled_placeholders}/{original_placeholders}")
    if remaining_placeholders > 0:
        print(f"⚠️  Remaining placeholders: {remaining_placeholders}")


if __name__ == "__main__":
    main()