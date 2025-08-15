#!/usr/bin/env python3
"""
LCLS Run Classifier with Context-Aware Chunking

This tool classifies LCLS experimental runs using chunking that builds output
incrementally. Each chunk includes preceding runs with their classifications
as context for better consistency and pattern recognition.
"""

import argparse
import json
import time
import os
import sys
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import logging


class StanfordAPIError(Exception):
    """Custom exception for Stanford API errors"""
    pass


class RunClassifier:
    """
    Client for Stanford AI Gateway API calls with context-aware chunking
    
    Builds output incrementally, chunk by chunk, with preceding runs providing
    rich context including their already-determined classifications.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-7-sonnet"):
        """
        Initialize the API client
        
        Args:
            api_key: Stanford API key (defaults to STANFORD_API_KEY env var)
            model: Model to use for analysis
        """
        self.api_key = api_key or os.getenv("STANFORD_API_KEY")
        if not self.api_key:
            raise StanfordAPIError("STANFORD_API_KEY environment variable not set")

        self.model = model
        self.base_url = "https://aiapi-prod.stanford.edu/v1"
        self.max_tokens = 8192
        self.temperature = 0.1

        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _make_api_call(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Make API call to Stanford AI Gateway with robust error handling"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Making API call (attempt {attempt + 1}/{max_retries})...")

                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=120
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    error_msg = f"API call failed with status {response.status_code}"
                    if response.text:
                        error_msg += f": {response.text}"

                    if attempt < max_retries - 1:
                        self.logger.warning(f"  {error_msg}, retrying...")
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        raise StanfordAPIError(error_msg)

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"  Request failed: {e}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise StanfordAPIError(f"Request failed after {max_retries} attempts: {e}")

        raise StanfordAPIError("Unexpected error in API call")

    def _validate_response(self, response: Dict[str, Any]) -> str:
        """Validate and extract content from API response"""
        try:
            content = response["choices"][0]["message"]["content"]
            if not content:
                raise StanfordAPIError("API returned empty content")
            return content.strip()
        except (KeyError, IndexError, TypeError) as e:
            raise StanfordAPIError(f"Invalid API response format: {e}")

    def _extract_json_from_response(self, content: str) -> str:
        """Extract JSON from API response content"""
        # Try to find JSON block in markdown code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Try to find standalone JSON object
        json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', content, re.DOTALL)
        if json_match:
            return json_match.group(1)
            
        return content

    def _validate_chunk_classifications(self, json_str: str) -> Dict[str, Any]:
        """Validate JSON output for chunk classifications"""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise StanfordAPIError(f"Invalid JSON response: {e}")

        # Check required top-level fields
        required_fields = ["experiment_id", "chunk_info", "classifications"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise StanfordAPIError(f"Missing required fields: {missing_fields}")

        # Validate classifications array
        if not isinstance(data["classifications"], list):
            raise StanfordAPIError("Classifications must be an array")
            
        # Check each classification entry
        valid_classifications = {"sample_run", "calibration_run", "alignment_run", "test_run", "commissioning_run", "unknown_run"}
        valid_confidence = {"high", "medium", "low"}
        
        for i, classification in enumerate(data["classifications"]):
            run_fields = ["run_number", "classification", "confidence", "key_evidence"]
            missing_run_fields = [field for field in run_fields if field not in classification]
            if missing_run_fields:
                raise StanfordAPIError(f"Run {i+1} missing required fields: {missing_run_fields}")
                
            if classification["classification"] not in valid_classifications:
                raise StanfordAPIError(f"Run {classification['run_number']} has invalid classification: {classification['classification']}")
                
            if classification["confidence"] not in valid_confidence:
                raise StanfordAPIError(f"Run {classification['run_number']} has invalid confidence: {classification['confidence']}")

        return data

    def _parse_experiment_file(self, markdown_content: str) -> Tuple[str, int, List[Dict]]:
        """
        Parse experiment file to extract experiment ID, run count, and run data
        
        Returns:
            Tuple of (experiment_id, total_runs, run_data_list)
        """
        # Extract experiment ID from header
        exp_match = re.search(r'# Experiment (\w+)', markdown_content)
        if not exp_match:
            raise StanfordAPIError("Could not extract experiment ID from file")
        experiment_id = exp_match.group(1)
        
        # Parse individual runs with their data
        run_pattern = r'### Run (\d+)\n(.*?)(?=### Run \d+|\Z)'
        runs = re.findall(run_pattern, markdown_content, re.DOTALL)
        
        run_data = []
        for run_num_str, run_content in runs:
            run_data.append({
                'run_number': int(run_num_str),
                'content': f"### Run {run_num_str}\n{run_content.strip()}"
            })
        
        return experiment_id, len(run_data), run_data

    def _read_previous_classifications(self, output_file: Path, num_context: int) -> List[Dict]:
        """
        Read the last N classifications from the output file
        
        Args:
            output_file: Path to the output file
            num_context: Number of previous classifications to read
            
        Returns:
            List of previous classification dictionaries
        """
        if not output_file.exists():
            return []
            
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                classifications = data.get('classifications', [])
                return classifications[-num_context:] if classifications else []
        except (json.JSONDecodeError, KeyError, IOError):
            return []

    def _create_context_content(self, context_classifications: List[Dict], raw_run_data: List[Dict]) -> str:
        """
        Create context content that includes previous classifications
        
        Args:
            context_classifications: Previous classification results
            raw_run_data: Raw run data for context runs
            
        Returns:
            Formatted context content with classifications included
        """
        if not context_classifications:
            return ""
            
        context_content = []
        
        # Create a map of run_number to classification for quick lookup
        classification_map = {c['run_number']: c for c in context_classifications}
        
        for run_data in raw_run_data:
            run_num = run_data['run_number']
            if run_num in classification_map:
                # Include the original content plus the classification
                classification = classification_map[run_num]
                
                # Replace __ANSWER__ placeholders with actual classifications
                content = run_data['content']
                content = content.replace(
                    "**Run classification**: __ANSWER__",
                    f"**Run classification**: {classification['classification']}"
                )
                content = content.replace(
                    "**Confidence**: __ANSWER__", 
                    f"**Confidence**: {classification['confidence']}"
                )
                content = content.replace(
                    "**Key evidence**: __ANSWER__",
                    f"**Key evidence**: {classification['key_evidence']}"
                )
                
                # Add context marker
                content = content.replace(
                    f"### Run {run_num}",
                    f"### Run {run_num} (CONTEXT - already classified)"
                )
                
                context_content.append(content)
        
        return "\n\n".join(context_content)

    def _create_prompt(self, experiment_id: str, context_content: str, 
                      new_runs: List[Dict], chunk_num: int, total_chunks: int) -> str:
        """Create classification prompt with rich context"""
        
        # Build new runs content (with __ANSWER__ placeholders)
        new_runs_content = "\n\n".join([run['content'] for run in new_runs])
        
        # Combine context and new content
        if context_content:
            full_content = f"{context_content}\n\n{new_runs_content}"
            context_note = f"\n\n**CONTEXT NOTE**: Runs marked as 'CONTEXT' are already classified and included for pattern recognition. Only classify the NEW runs ({new_runs[0]['run_number']}-{new_runs[-1]['run_number']})."
        else:
            full_content = new_runs_content
            context_note = ""

        new_run_numbers = [run['run_number'] for run in new_runs]
        classify_start = new_run_numbers[0]
        classify_end = new_run_numbers[-1]

        prompt = f"""# Experiment {experiment_id} - Run Classification (Chunk {chunk_num}/{total_chunks})

{full_content}

# LCLS Run Classification Expert

You are a scientific data analyst specializing in LCLS (Linac Coherent Light Source) experiments. Your task is to analyze logbook entries from experimental runs and classify each run based on its primary purpose and activities.{context_note}

## Classification Framework

Classify each run into exactly ONE of these categories based on its PRIMARY purpose:

### 1. `sample_run`
**Definition**: Runs where real biological, chemical, or material samples are measured/analyzed for scientific data collection.

**Key Indicators**:
- Sample names, concentrations, or chemical formulas (e.g., "Fe(bpy)3", "10mM protein", "AgB")
- Sample delivery mentions (injection rates, flow rates, pressure values)
- Data collection on actual samples
- Sample quality assessments ("good sample position", "out of sample")
- Scientific measurement parameters
- Sample preparation activities leading to measurement

### 2. `calibration_run`
**Definition**: Runs focused on detector calibration, dark measurements, or establishing baseline conditions.

**Key Indicators**:
- "DARK" entries (with or without capitalization)
- Detector calibration activities ("pedestal", "gain settings")
- Background measurements without samples
- "takepeds", "makepeds" commands
- Detector bad pixel analysis
- Baseline establishment
- Energy calibration (notch scans, monochromator adjustments)

### 3. `alignment_run`
**Definition**: Runs dedicated to beam alignment, optical positioning, or spatial calibration.

**Key Indicators**:
- Beam pointing/positioning ("beam alignment", "mirror positions")
- YAG screen usage for alignment
- Motor positioning activities
- Focus adjustments and optimization
- Mirror/optics positioning
- Wire scans for beam characterization
- Spatial calibration activities

### 4. `test_run`
**Definition**: Runs for equipment testing, troubleshooting, or system verification (not including commissioning).

**Key Indicators**:
- Equipment testing ("injector testing", "testing PSL spheres")
- Troubleshooting activities
- System verification without samples
- Performance testing
- "test" or "testing" explicitly mentioned (without commissioning context)

### 5. `commissioning_run`
**Definition**: Runs for instrument commissioning, initial setup, or end station preparation.

**Key Indicators**:
- "commissioning", "commission", "checkout" mentions
- Initial instrument setup
- End station preparation
- System bring-up activities
- Machine development (MD) activities

### 6. `unknown_run`
**Definition**: Runs with insufficient, unclear, or contradictory information to classify confidently.

## Confidence Guidelines

- **high**: Clear, unambiguous indicators with multiple supporting evidence
- **medium**: Some clear indicators but with minor ambiguity, mixed activities but one primary purpose apparent
- **low**: Limited or unclear evidence, contradictory activities, classification based on weak indicators

## Classification Priority Rules

When a run contains multiple types of activities, classify based on the PRIMARY purpose:

1. **Sample measurement** takes priority over setup activities
2. **Calibration** takes priority over routine maintenance
3. **Alignment** takes priority over general testing
4. **Commissioning** applies only to dedicated commissioning runs
5. **Use `unknown_run`** only when truly unclear

## Scientific Domain Knowledge

- **LCLS instruments**: AMO, CXI, MFX, MEC, XPP, XCS, RIX, TMO
- **Common samples**: Proteins, crystals, foils, gases, liquids, nanoparticles
- **Measurement types**: Diffraction, spectroscopy, imaging, scattering
- **Equipment**: Detectors, motors, mirrors, injectors, YAG screens

## Required Output (JSON only):

{{
  "experiment_id": "{experiment_id}",
  "chunk_info": {{
    "chunk_number": {chunk_num},
    "total_chunks": {total_chunks},
    "classify_start": {classify_start},
    "classify_end": {classify_end}
  }},
  "classifications": [
    {{
      "run_number": {classify_start},
      "classification": "calibration_run",
      "confidence": "high",
      "key_evidence": "DARK measurement and detector bad pixel analysis"
    }}
    // Continue for runs {classify_start} through {classify_end} ONLY
  ]
}}

IMPORTANT: Classify ONLY the NEW runs ({classify_start} through {classify_end}). Do not re-classify context runs. Return ONLY the JSON object with no additional text."""

        return prompt

    def _initialize_output_file(self, output_file: Path, experiment_id: str) -> None:
        """Initialize the output file with metadata"""
        initial_data = {
            "experiment_id": experiment_id,
            "total_runs": 0,
            "processing_info": {
                "processing_mode": "chunked_classification",
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "chunks_completed": 0
            },
            "classifications": []
        }
        
        with open(output_file, 'w') as f:
            json.dump(initial_data, f, indent=2)

    def _append_classifications(self, output_file: Path, new_classifications: List[Dict], 
                              chunk_num: int, total_chunks: int) -> None:
        """Append new classifications to the output file"""
        # Read current data
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Append new classifications
        data["classifications"].extend(new_classifications)
        data["total_runs"] = len(data["classifications"])
        data["processing_info"]["chunks_completed"] = chunk_num
        data["processing_info"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        
        if chunk_num == total_chunks:
            data["processing_info"]["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        
        # Write back to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

    def classify_runs(self, markdown_content: str, chunk_size: int = 30, 
                     context_runs: int = 5, rate_limit: float = 1.0,
                     output_file: Optional[Path] = None,
                     resume_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Classify runs using chunking with context from preceding runs
        
        Args:
            markdown_content: Markdown-formatted experiment data
            chunk_size: Number of runs to classify per chunk
            context_runs: Number of preceding runs to include for context
            rate_limit: Delay between API calls in seconds
            output_file: Path for output
            resume_file: Path for resume data (if enabled)
            
        Returns:
            Final classification results dictionary
        """
        # Parse experiment
        experiment_id, total_runs, run_data = self._parse_experiment_file(markdown_content)
        
        self.logger.info(f"Processing experiment {experiment_id} with {total_runs} runs")
        self.logger.info(f"Chunking: {chunk_size} runs per chunk, {context_runs} preceding runs for context")
        
        # Calculate chunks
        total_chunks = (total_runs + chunk_size - 1) // chunk_size  # Ceiling division
        self.logger.info(f"Will process {total_chunks} chunks")
        
        # Initialize output file
        if not output_file:
            output_file = Path(f"{experiment_id}_classifications.json")
        
        # Check for resume capability
        start_chunk = 1
        if resume_file and resume_file.exists():
            try:
                with open(resume_file, 'r') as f:
                    resume_data = json.load(f)
                    start_chunk = resume_data.get('last_completed_chunk', 0) + 1
                    self.logger.info(f"Resuming from chunk {start_chunk}/{total_chunks}")
            except Exception as e:
                self.logger.warning(f"Could not load resume data: {e}, starting from beginning")
        
        if start_chunk == 1:
            self._initialize_output_file(output_file, experiment_id)
        
        # Process chunks
        with tqdm(total=total_chunks, initial=start_chunk-1, desc="Processing chunks") as pbar:
            for chunk_num in range(start_chunk, total_chunks + 1):
                chunk_start_time = time.time()
                
                try:
                    # Calculate chunk boundaries
                    run_start_idx = (chunk_num - 1) * chunk_size
                    run_end_idx = min(run_start_idx + chunk_size, total_runs)
                    
                    # Get new runs to classify
                    new_runs = run_data[run_start_idx:run_end_idx]
                    
                    # Get context runs (preceding runs with their classifications)
                    context_content = ""
                    if context_runs > 0 and chunk_num > 1:
                        # Get previous classifications for context
                        previous_classifications = self._read_previous_classifications(output_file, context_runs)
                        
                        # Get corresponding raw run data for context
                        context_start_idx = max(0, run_start_idx - context_runs)
                        context_run_data = run_data[context_start_idx:run_start_idx]
                        
                        context_content = self._create_context_content(previous_classifications, context_run_data)
                    
                    # Create prompt
                    prompt = self._create_prompt(
                        experiment_id, context_content, new_runs, chunk_num, total_chunks
                    )
                    
                    # Make API call
                    self.logger.debug(f"Processing chunk {chunk_num}/{total_chunks}: runs {new_runs[0]['run_number']}-{new_runs[-1]['run_number']}")
                    response = self._make_api_call(prompt)
                    
                    # Extract and validate
                    content = self._validate_response(response)
                    json_content = self._extract_json_from_response(content)
                    result = self._validate_chunk_classifications(json_content)
                    
                    # Append to output
                    self._append_classifications(output_file, result["classifications"], chunk_num, total_chunks)
                    
                    # Save resume data
                    if resume_file:
                        resume_data = {
                            'experiment_id': experiment_id,
                            'total_chunks': total_chunks,
                            'last_completed_chunk': chunk_num,
                            'last_updated': time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                        }
                        with open(resume_file, 'w') as f:
                            json.dump(resume_data, f, indent=2)
                    
                    # User feedback
                    chunk_time = time.time() - chunk_start_time
                    classified_count = len(result["classifications"])
                    start_run = new_runs[0]['run_number']
                    end_run = new_runs[-1]['run_number']
                    
                    print(f"✓ Chunk {chunk_num}/{total_chunks} complete: classified runs {start_run}-{end_run} ({classified_count} runs) in {chunk_time:.1f}s", file=sys.stderr)
                    
                    # Rate limiting
                    if rate_limit > 0 and chunk_num < total_chunks:
                        time.sleep(rate_limit)
                        
                    pbar.update(1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_num}: {e}")
                    raise StanfordAPIError(f"Chunk {chunk_num} failed: {e}")
        
        # Load final result
        with open(output_file, 'r') as f:
            final_result = json.load(f)
        
        # Clean up resume file if successful
        if resume_file and resume_file.exists():
            resume_file.unlink()
            
        self.logger.info(f"Classification complete: {final_result['total_runs']} runs classified")
        return final_result


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="LCLS Run Classifier with Context-Aware Chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_classifier.py experiment.md
  
  # Custom chunk size and context
  python run_classifier.py experiment.md --chunk-size 20 --context-runs 3
  
  # With resume capability
  python run_classifier.py experiment.md --resume --rate-limit 2.0
  
  # Verbose output with custom settings
  python run_classifier.py experiment.md -v --chunk-size 25 --context-runs 5
        """
    )
    
    # Required arguments
    parser.add_argument('input_file', type=Path,
                       help='Input markdown file with experiment data')
    
    # Output options
    parser.add_argument('-o', '--output', type=Path,
                       help='Output JSON file (default: auto-generated from input)')
    
    # Chunking options
    parser.add_argument('--chunk-size', type=int, default=30,
                       help='Number of runs to classify per chunk (default: 30)')
    parser.add_argument('--context-runs', type=int, default=5,
                       help='Number of preceding runs to include for context (default: 5)')
    
    # API options
    parser.add_argument('--api-key', 
                       help='Stanford API key (default: STANFORD_API_KEY env var)')
    parser.add_argument('--model', default='claude-3-7-sonnet',
                       help='Model to use (default: claude-3-7-sonnet)')
    parser.add_argument('--rate-limit', type=float, default=1.0,
                       help='Delay between API calls in seconds (default: 1.0)')
    
    # Resume options
    parser.add_argument('--resume', action='store_true',
                       help='Enable resume capability for interrupted processing')
    parser.add_argument('--resume-file', type=Path,
                       help='File for storing resume data (default: auto-generated)')
    
    # Logging options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input file
        if not args.input_file.exists():
            logger.error(f"Input file not found: {args.input_file}")
            sys.exit(1)
        
        # Generate output filename if not provided
        if not args.output:
            base_name = args.input_file.stem
            if base_name.endswith('_enrichment'):
                base_name = base_name[:-11]
            args.output = Path(f"{base_name}_classifications.json")
        
        # Generate resume filename if resume enabled but not specified
        if args.resume and not args.resume_file:
            args.resume_file = args.output.with_suffix('.resume.json')
        
        # Read input file
        logger.info(f"Reading input file: {args.input_file}")
        with open(args.input_file, 'r') as f:
            markdown_content = f.read()
        
        # Create classifier
        classifier = RunClassifier(api_key=args.api_key, model=args.model)
        
        # Process with chunking
        logger.info("Starting classification...")
        result = classifier.classify_runs(
            markdown_content=markdown_content,
            chunk_size=args.chunk_size,
            context_runs=args.context_runs,
            rate_limit=args.rate_limit,
            output_file=args.output,
            resume_file=args.resume_file if args.resume else None
        )
        
        print(f"\n✓ Classification complete!", file=sys.stderr)
        print(f"✓ Experiment: {result['experiment_id']}", file=sys.stderr)
        print(f"✓ Total runs: {result['total_runs']}", file=sys.stderr)
        print(f"✓ Chunks processed: {result['processing_info']['chunks_completed']}", file=sys.stderr)
        print(f"✓ Output saved to: {args.output}", file=sys.stderr)
        
        # Print classification breakdown
        classification_counts = {}
        for classification in result['classifications']:
            cat = classification['classification']
            classification_counts[cat] = classification_counts.get(cat, 0) + 1
            
        print("✓ Classification breakdown:", file=sys.stderr)
        for cat, count in sorted(classification_counts.items()):
            print(f"  - {cat}: {count}", file=sys.stderr)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        if args.resume and args.resume_file and args.resume_file.exists():
            logger.info(f"Resume data saved to: {args.resume_file}")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()