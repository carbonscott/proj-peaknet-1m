#!/usr/bin/env python3
"""
Batch Preprocessing for All LCLS Experiments

This script processes all experiments in the database and generates
LLM-ready markdown files for metadata enrichment.

Features:
- Parallel processing using joblib for faster execution
- Batch processing with configurable batch sizes
- Resume capability to skip already processed experiments
- Comprehensive error logging and progress tracking

Usage:
    # Basic usage with default parallel processing
    python batch_preprocess_logbook.py --output-dir processed_experiments

    # Resume processing with 8 parallel jobs
    python batch_preprocess_logbook.py --output-dir processed_experiments --resume --n-jobs 8

    # Test run with limited experiments
    python batch_preprocess_logbook.py --limit 10 --output-dir test_run --n-jobs 4

    # Sequential processing (for debugging)
    python batch_preprocess_logbook.py --output-dir processed_experiments --n-jobs 1

    # Custom batch size for memory management
    python batch_preprocess_logbook.py --output-dir processed_experiments --batch-size 20
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback
import time
from multiprocessing import cpu_count

# Parallel processing
from joblib import Parallel, delayed

# Import our preprocessing class
from preprocess_logbook import LogbookPreprocessor


def process_experiment_worker(experiment_info: Dict, db_path: str, output_dir: str) -> Tuple[str, bool, Optional[str]]:
    """
    Worker function to process a single experiment in parallel.

    Args:
        experiment_info: Dictionary containing experiment metadata
        db_path: Path to the database file
        output_dir: Directory to write output files

    Returns:
        Tuple of (experiment_id, success, error_message)
    """
    experiment_id = experiment_info['experiment_id']
    output_dir = Path(output_dir)

    try:
        # Initialize preprocessor with its own database connection
        preprocessor = LogbookPreprocessor(db_path)
        preprocessor.connect()

        # Process experiment
        result = preprocessor.process_experiment(experiment_id)

        # Check if we got any data
        if "No logbook data found" in result:
            error_msg = "No logbook data found"
            preprocessor.close()
            return experiment_id, False, error_msg

        # Write output file
        output_file = output_dir / f"{experiment_id}_enrichment.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)

        # Add metadata comment to track processing
        metadata = {
            'processed_at': datetime.now().isoformat(),
            'total_runs': experiment_info['total_runs'],
            'total_logbook_entries': experiment_info['total_logbook_entries'],
            'instrument': experiment_info['instrument']
        }

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n<!-- Processing metadata: {json.dumps(metadata)} -->\n")

        # Clean up
        preprocessor.close()

        return experiment_id, True, None

    except Exception as e:
        error_msg = f"{str(e)} - {traceback.format_exc()}"
        return experiment_id, False, error_msg


class BatchProcessor:
    """Batch processor for all LCLS experiments."""

    def __init__(self, db_path: str, output_dir: str, n_jobs: int = 1, batch_size: int = 50):
        """Initialize batch processor."""
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.stats_file = self.output_dir / "batch_stats.json"
        self.error_log = self.output_dir / "errors.log"

        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)

        # Load existing stats if resuming
        self.stats = self.load_stats()

        # Initialize database connection
        self.conn = None

    def connect(self):
        """Connect to database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def load_stats(self) -> Dict:
        """Load existing batch processing statistics."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        return {
            'start_time': None,
            'end_time': None,
            'total_experiments': 0,
            'processed_experiments': 0,
            'successful_experiments': 0,
            'failed_experiments': 0,
            'skipped_experiments': 0,
            'errors': [],
            'processed_list': [],
            'failed_list': []
        }

    def save_stats(self):
        """Save current statistics."""
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def log_error(self, experiment_id: str, error: str):
        """Log an error to the error log file."""
        timestamp = datetime.now().isoformat()
        with open(self.error_log, 'a') as f:
            f.write(f"[{timestamp}] {experiment_id}: {error}\\n")

        # Also add to stats
        self.stats['errors'].append({
            'timestamp': timestamp,
            'experiment_id': experiment_id,
            'error': error
        })
        self.stats['failed_list'].append(experiment_id)

    def get_all_experiments(self, limit: Optional[int] = None) -> List[Dict]:
        """Get all experiments with logbook data."""
        query = """
        SELECT e.experiment_id, e.instrument, 
               COUNT(DISTINCT r.run_number) as total_runs,
               COUNT(l.log_id) as total_logbook_entries,
               MIN(l.timestamp) as first_entry,
               MAX(l.timestamp) as last_entry
        FROM Experiment e 
        LEFT JOIN Run r ON e.experiment_id = r.experiment_id 
        LEFT JOIN Logbook l ON r.run_id = l.run_id 
        WHERE l.content IS NOT NULL 
        GROUP BY e.experiment_id 
        ORDER BY total_logbook_entries DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        cursor = self.conn.execute(query)
        rows = cursor.fetchall()

        experiments = []
        for row in rows:
            experiments.append({
                'experiment_id': row['experiment_id'],
                'instrument': row['instrument'],
                'total_runs': row['total_runs'],
                'total_logbook_entries': row['total_logbook_entries'],
                'first_entry': row['first_entry'],
                'last_entry': row['last_entry']
            })

        return experiments

    def get_output_file(self, experiment_id: str) -> Path:
        """Get output file path for an experiment."""
        return self.output_dir / f"{experiment_id}_enrichment.md"

    def is_already_processed(self, experiment_id: str) -> bool:
        """Check if an experiment has already been processed."""
        output_file = self.get_output_file(experiment_id)
        return output_file.exists() and experiment_id in self.stats.get('processed_list', [])


    def process_all_experiments(self, limit: Optional[int] = None, resume: bool = False):
        """Process all experiments with parallel processing and progress tracking."""
        print(f"Starting batch processing...")
        print(f"Database: {self.db_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Resume mode: {resume}")
        print(f"Parallel jobs: {self.n_jobs}")
        print(f"Batch size: {self.batch_size}")

        # Get all experiments
        experiments = self.get_all_experiments(limit)
        total_experiments = len(experiments)

        print(f"Found {total_experiments} experiments with logbook data")

        # Filter out already processed experiments if resuming
        if resume:
            experiments_to_process = []
            skipped_count = 0
            for exp_info in experiments:
                if self.is_already_processed(exp_info['experiment_id']):
                    skipped_count += 1
                else:
                    experiments_to_process.append(exp_info)
            experiments = experiments_to_process
            print(f"Skipping {skipped_count} already processed experiments")
            print(f"Processing {len(experiments)} remaining experiments")
        else:
            skipped_count = 0

        if not experiments:
            print("No experiments to process!")
            return

        # Update stats
        self.stats['total_experiments'] = total_experiments
        if not self.stats['start_time']:
            self.stats['start_time'] = datetime.now().isoformat()

        # Process experiments in batches
        successful_count = 0
        failed_count = 0
        processed_count = 0

        # Split experiments into batches
        batches = [experiments[i:i + self.batch_size] for i in range(0, len(experiments), self.batch_size)]
        total_batches = len(batches)

        print(f"Processing in {total_batches} batches of up to {self.batch_size} experiments each")

        for batch_idx, batch in enumerate(batches, 1):
            batch_start_time = time.time()

            print(f"\\nProcessing batch {batch_idx}/{total_batches} ({len(batch)} experiments)...")

            if self.n_jobs == 1:
                # Sequential processing for debugging or when n_jobs=1
                results = []
                for exp_info in batch:
                    result = process_experiment_worker(exp_info, self.db_path, str(self.output_dir))
                    results.append(result)
            else:
                # Parallel processing
                results = Parallel(n_jobs=self.n_jobs, backend='loky')(
                    delayed(process_experiment_worker)(exp_info, self.db_path, str(self.output_dir))
                    for exp_info in batch
                )

            # Process results from this batch
            batch_successful = 0
            batch_failed = 0

            for experiment_id, success, error_msg in results:
                processed_count += 1

                if success:
                    successful_count += 1
                    batch_successful += 1
                    # Add to processed list
                    if experiment_id not in self.stats['processed_list']:
                        self.stats['processed_list'].append(experiment_id)
                else:
                    failed_count += 1
                    batch_failed += 1
                    # Log error
                    self.log_error(experiment_id, error_msg)

            # Batch timing and progress
            batch_time = time.time() - batch_start_time
            overall_progress = (batch_idx / total_batches) * 100

            print(f"Batch {batch_idx} completed in {batch_time:.1f}s: "
                  f"{batch_successful} successful, {batch_failed} failed")
            print(f"Overall progress: {overall_progress:.1f}% "
                  f"({processed_count}/{len(experiments)} experiments)")

            # Update stats periodically
            self.stats['processed_experiments'] = processed_count
            self.stats['successful_experiments'] = successful_count
            self.stats['failed_experiments'] = failed_count
            self.stats['skipped_experiments'] = skipped_count
            self.save_stats()

        # Final stats update
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['processed_experiments'] = processed_count
        self.stats['successful_experiments'] = successful_count
        self.stats['failed_experiments'] = failed_count
        self.stats['skipped_experiments'] = skipped_count
        self.save_stats()

        # Final report
        print(f"\\n\\nBatch processing completed!")
        print(f"Total experiments: {total_experiments}")
        print(f"Processed: {processed_count}")
        print(f"Successful: {successful_count}")
        print(f"Failed: {failed_count}")
        print(f"Skipped: {skipped_count}")

        if failed_count > 0:
            print(f"\\nErrors logged to: {self.error_log}")

        print(f"Statistics saved to: {self.stats_file}")
        print(f"Output files in: {self.output_dir}")

    def generate_summary_report(self) -> str:
        """Generate a summary report of batch processing."""
        if not self.stats['end_time']:
            return "Batch processing not completed yet."

        # Calculate processing time
        start_time = datetime.fromisoformat(self.stats['start_time'])
        end_time = datetime.fromisoformat(self.stats['end_time'])
        duration = end_time - start_time

        report = []
        report.append("# Batch Processing Summary Report")
        report.append(f"")
        report.append(f"**Processing Time**: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Duration**: {duration}")
        report.append(f"")
        report.append(f"## Configuration")
        report.append(f"- **Parallel jobs**: {self.n_jobs}")
        report.append(f"- **Batch size**: {self.batch_size}")
        report.append(f"")
        report.append(f"## Statistics")
        report.append(f"- **Total experiments**: {self.stats['total_experiments']}")
        report.append(f"- **Processed**: {self.stats['processed_experiments']}")
        report.append(f"- **Successful**: {self.stats['successful_experiments']}")
        report.append(f"- **Failed**: {self.stats['failed_experiments']}")
        report.append(f"- **Skipped**: {self.stats['skipped_experiments']}")
        report.append(f"")

        # Success rate
        if self.stats['processed_experiments'] > 0:
            success_rate = (self.stats['successful_experiments'] / self.stats['processed_experiments']) * 100
            report.append(f"**Success Rate**: {success_rate:.1f}%")

        # Error summary
        if self.stats['failed_experiments'] > 0:
            report.append(f"")
            report.append(f"## Failed Experiments")
            for failed_exp in self.stats['failed_list'][-10:]:  # Show last 10 failures
                report.append(f"- {failed_exp}")

            if len(self.stats['failed_list']) > 10:
                report.append(f"- ... and {len(self.stats['failed_list']) - 10} more")

        return "\\n".join(report)


def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(description='Batch process all LCLS experiments')
    parser.add_argument('--database', default='2025_0813_2257.db', 
                       help='Database file path (default: 2025_0813_2257.db)')
    parser.add_argument('--output-dir', default='processed_experiments',
                       help='Output directory for processed files (default: processed_experiments)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of experiments to process (for testing)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume processing, skip already processed experiments')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate summary report only, do not process')
    parser.add_argument('--n-jobs', type=int, default=cpu_count(),
                       help=f'Number of parallel jobs (default: {cpu_count()}, use 1 for sequential)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of experiments to process in each batch (default: 50)')

    args = parser.parse_args()

    # Initialize batch processor
    processor = BatchProcessor(args.database, args.output_dir, args.n_jobs, args.batch_size)

    try:
        if args.report_only:
            # Generate report only
            report = processor.generate_summary_report()
            print(report)
        else:
            # Connect and process
            processor.connect()
            processor.process_all_experiments(args.limit, args.resume)

            # Generate summary report
            report = processor.generate_summary_report()
            report_file = Path(args.output_dir) / "summary_report.md"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"\\nSummary report saved to: {report_file}")

    finally:
        processor.close()


if __name__ == "__main__":
    main()
