# LCLS Run Classification Pipeline

AI-powered classification of LCLS experimental runs from logbook entries.

## 🔄 Core Pipeline

```
SQLite Database (logbook entries)
         ↓
   preprocess_logbook.py (batch_preprocess_logbook.py for bulk)
         ↓
*_enrichment.md (with __ANSWER__ placeholders)
         ↓
   run_classifier.py (batch_run_classifier.py for bulk)
         ↓
*_classifications.json
         ↓
   fill_enrichment.py (batch_fill_enrichment.py for bulk)
         ↓
*_full_enrichment.md (complete)
```

## 📋 Core Scripts

**Data Preprocessing:**
- `preprocess_logbook.py` - Extract logbook entries from SQLite database
- `batch_preprocess_logbook.py` - Bulk preprocessing with parallel processing

**Classification:**
- `run_classifier.py` - Classify runs using AI (6 categories: sample, calibration, alignment, test, commissioning, unknown)
- `batch_run_classifier.py` - Process multiple experiments from CSV

**Enrichment:**
- `fill_enrichment.py` - Fill __ANSWER__ placeholders with classifications
- `batch_fill_enrichment.py` - Bulk placeholder filling

**Analysis:**
- `token_summary.py` - Analyze token usage and costs

## 🚀 Quick Start

```bash
# Single experiment
python run_classifier.py processed_experiments/mfx101080524_enrichment.md

# Batch processing  
python batch_run_classifier.py crystallography.csv

# Fill placeholders
python batch_fill_enrichment.py
```

## 📁 Files to Include in Git

**Core scripts**: `preprocess_logbook.py`, `batch_preprocess_logbook.py`, `run_classifier.py`, `batch_run_classifier.py`, `fill_enrichment.py`, `batch_fill_enrichment.py`, `token_summary.py`

**Config**: `crystallography.csv`

**Exclude**: Generated outputs (`batch_*_results/`, `fully_enriched_experiments/`, `processed_experiments/`)