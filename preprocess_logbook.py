#!/usr/bin/env python3
"""
Logbook Data Preprocessing Pipeline

This script processes logbook entries from LCLS experiments to prepare them for LLM-based
metadata enrichment. It groups entries by run, cleans HTML content, and formats the data
for run classification tasks.

Usage:
    python preprocess_logbook.py --experiment mfxl1027922 --output processed_data.md
"""

import sqlite3
import re
import html
import argparse
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class LogbookPreprocessor:
    """Preprocesses LCLS experiment logbook data for LLM enrichment."""

    def __init__(self, db_path: str, filter_patterns: Optional[List[str]] = None):
        """Initialize with database path and optional filter patterns."""
        self.db_path = db_path
        self.conn = None
        self.filter_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in (filter_patterns or [])]

    def connect(self):
        """Connect to SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def extract_experiment_data(self, experiment_id: str) -> Dict[int, List[Dict]]:
        """
        Extract all logbook entries for an experiment, grouped by run number.

        Args:
            experiment_id: The experiment identifier (e.g., 'mfxl1027922')

        Returns:
            Dictionary mapping run_number to list of logbook entries
        """
        query = """
        SELECT r.run_number, r.start_time, r.end_time, l.timestamp, l.content, l.author
        FROM Run r 
        LEFT JOIN Logbook l ON r.run_id = l.run_id 
        WHERE r.experiment_id = ? 
        AND l.content IS NOT NULL 
        AND LENGTH(TRIM(l.content)) > 0
        ORDER BY r.run_number, l.timestamp
        """

        cursor = self.conn.execute(query, (experiment_id,))
        rows = cursor.fetchall()

        runs_data = defaultdict(list)

        for row in rows:
            run_num = row['run_number']
            entry = {
                'run_number': run_num,
                'run_start': row['start_time'],
                'run_end': row['end_time'],
                'timestamp': row['timestamp'],
                'content': row['content'],
                'author': row['author']
            }
            runs_data[run_num].append(entry)

        return dict(runs_data)

    def clean_html_content(self, content: str) -> str:
        """
        Remove all HTML content and clean up text formatting.

        Args:
            content: Raw logbook entry content

        Returns:
            Cleaned text content
        """
        if not content:
            return ""

        # Decode HTML entities
        content = html.unescape(content)

        # Convert HTML tables to simple text format
        content = self._convert_html_tables(content)

        # Remove all HTML tags
        content = re.sub(r'<[^>]+>', '', content)

        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()

        # Remove very long technical logs (>1500 chars) but preserve short summary
        if len(content) > 1500:
            # Try to extract meaningful summary from start
            lines = content.split('\n')
            summary_lines = []
            for line in lines[:10]:  # Take first 10 lines
                line = line.strip()
                if line and not line.startswith(('ERROR:', 'DEBUG:', 'INFO:')):
                    summary_lines.append(line)
                if len(' '.join(summary_lines)) > 200:
                    break

            if summary_lines:
                content = ' '.join(summary_lines) + " [Long technical log truncated]"
            else:
                content = "[Long technical log - content truncated]"

        return content

    def should_filter_entry(self, content: str) -> bool:
        """Check if an entry should be filtered out based on configured patterns."""
        if not self.filter_patterns or not content:
            return False

        # Check if content matches any filter pattern
        for pattern in self.filter_patterns:
            if pattern.search(content.strip()):
                return True
        return False

    def _convert_html_tables(self, content: str) -> str:
        """Convert HTML tables to markdown-style format."""
        # Simple table conversion - extract cell contents
        table_pattern = r'<table[^>]*>(.*?)</table>'

        def convert_table(match):
            table_content = match.group(1)

            # Extract table cells
            cell_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'
            cells = re.findall(cell_pattern, table_content, re.DOTALL | re.IGNORECASE)

            if not cells:
                return "[Table content]"

            # Clean cells and create simple format
            cleaned_cells = []
            for cell in cells:
                cell_text = re.sub(r'<[^>]+>', '', cell).strip()
                if cell_text:
                    cleaned_cells.append(cell_text)

            if cleaned_cells:
                return " | ".join(cleaned_cells[:6])  # Limit to 6 columns
            return "[Table content]"

        return re.sub(table_pattern, convert_table, content, flags=re.DOTALL | re.IGNORECASE)

    def deduplicate_entries(self, entries: List[Dict]) -> List[Dict]:
        """
        Remove duplicate entries while preserving unique information.

        Args:
            entries: List of logbook entries for a run

        Returns:
            Deduplicated list of entries
        """
        seen_content = set()
        unique_entries = []
        duplicate_counts = defaultdict(int)

        for entry in entries:
            content = entry['content'].strip()

            # Skip empty content
            if not content:
                continue

            # Track duplicates
            if content in seen_content:
                duplicate_counts[content] += 1
                continue

            seen_content.add(content)
            unique_entries.append(entry)

        # Add summary of duplicates for significant ones
        for content, count in duplicate_counts.items():
            if count > 2:  # Only note if more than 2 duplicates
                summary_entry = {
                    'content': f"[Repeated {count} times: {content[:50]}...]",
                    'timestamp': unique_entries[-1]['timestamp'] if unique_entries else None,
                    'author': 'system'
                }
                unique_entries.append(summary_entry)

        return unique_entries

    def create_run_context(self, run_number: int, entries: List[Dict]) -> str:
        """
        Create a formatted context block for a single run.

        Args:
            run_number: The run number
            entries: List of logbook entries for this run

        Returns:
            Formatted markdown context for the run
        """
        if not entries:
            return f"### Run {run_number}\n**Status**: No logbook entries\n\n"

        # Get run metadata
        first_entry = entries[0]
        run_start = first_entry.get('run_start', 'Unknown')
        run_end = first_entry.get('run_end', 'Unknown')

        # Calculate duration if possible
        duration = "Unknown"
        if run_start and run_end and run_start != 'Unknown' and run_end != 'Unknown':
            try:
                start_dt = datetime.fromisoformat(run_start.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(run_end.replace('Z', '+00:00'))
                duration_sec = (end_dt - start_dt).total_seconds()
                if duration_sec < 60:
                    duration = f"{duration_sec:.1f} seconds"
                elif duration_sec < 3600:
                    duration = f"{duration_sec/60:.1f} minutes"
                else:
                    duration = f"{duration_sec/3600:.1f} hours"
            except:
                duration = "Unknown"

        # Deduplicate entries
        unique_entries = self.deduplicate_entries(entries)

        # Create context block
        context = f"### Run {run_number}\n"
        context += f"**Duration**: {duration}\n"
        context += f"**Total entries**: {len(entries)} ({len(unique_entries)} unique)\n"
        context += "**Activities**:\n"

        filtered_count = 0
        for entry in unique_entries:
            cleaned_content = self.clean_html_content(entry['content'])
            if cleaned_content and len(cleaned_content) > 3:
                # Check if this entry should be filtered out
                if self.should_filter_entry(cleaned_content):
                    filtered_count += 1
                    continue

                # Add bullet point and truncate if still too long
                if len(cleaned_content) > 200:
                    cleaned_content = cleaned_content[:200] + "..."
                context += f"- {cleaned_content}\n"

        # Add filtering summary if entries were filtered
        if filtered_count > 0:
            context += f"- [Filtered out {filtered_count} automated log entries]\n"

        # Add enrichment template fields for LLM processing
        context += "\n**Run classification**: __ANSWER__\n"
        context += "**Confidence**: __ANSWER__\n"
        context += "**Key evidence**: __ANSWER__\n"
        context += "\n"
        return context

    def process_experiment(self, experiment_id: str) -> str:
        """
        Process all runs for an experiment and create LLM-ready output.

        Args:
            experiment_id: The experiment identifier

        Returns:
            Formatted markdown content ready for LLM processing
        """
        # Extract data
        runs_data = self.extract_experiment_data(experiment_id)

        if not runs_data:
            return f"# Experiment {experiment_id}\n\nNo logbook data found.\n"

        # Create header
        output = f"# Experiment {experiment_id} - Logbook Analysis\n\n"
        output += f"**Total runs with logbook entries**: {len(runs_data)}\n\n"
        output += "## Run-by-Run Activities\n\n"

        # Process each run
        for run_number in sorted(runs_data.keys()):
            entries = runs_data[run_number]
            run_context = self.create_run_context(run_number, entries)
            output += run_context

        return output


def main():
    """Main entry point for the preprocessing script."""
    parser = argparse.ArgumentParser(description='Preprocess LCLS logbook data for LLM enrichment')
    parser.add_argument('--experiment', required=True, help='Experiment ID (e.g., mfxl1027922)')
    parser.add_argument('--database', default='2025_0813_2257.db', help='Database file path')
    parser.add_argument('--output', help='Output file path (default: stdout)')

    # Filtering options
    parser.add_argument('--filter-pattern', type=str, action='append', help='Single regex pattern to filter out (can be used multiple times)')
    parser.add_argument('--filter-patterns', type=str, help='File containing custom regex patterns to filter out (one per line)')

    args = parser.parse_args()

    # Initialize preprocessor with filtering options
    filter_patterns = []

    # Collect patterns from single --filter-pattern arguments
    if args.filter_pattern:
        filter_patterns.extend(args.filter_pattern)

    # Collect patterns from --filter-patterns file
    if args.filter_patterns:
        try:
            with open(args.filter_patterns, 'r') as f:
                file_patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                filter_patterns.extend(file_patterns)
        except FileNotFoundError:
            print(f"Warning: Filter patterns file '{args.filter_patterns}' not found. Skipping file-based patterns.")

    preprocessor = LogbookPreprocessor(args.database, filter_patterns=filter_patterns)

    try:
        preprocessor.connect()

        # Process experiment
        result = preprocessor.process_experiment(args.experiment)

        # Output result
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Processed data written to {args.output}")
        else:
            print(result)

    finally:
        preprocessor.close()


if __name__ == "__main__":
    main()
