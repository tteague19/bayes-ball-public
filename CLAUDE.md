# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bayes Ball is a Bayesian machine learning project for predicting football match outcomes using probabilistic modeling. The system combines TensorFlow Probability with MLflow experiment tracking to build and evaluate Bayesian Neural Networks for sports prediction with uncertainty quantification.

## Common Commands

### Environment Setup
```bash
# Install dependencies
poetry install

# Code formatting and linting
poetry run black .
poetry run flake8
poetry run isort .
```

### Data Ingestion
```bash
# Ingest football-data.co.uk data
bash scripts/ingestion/run_ingest_fduk.sh

# Or run directly with custom parameters
poetry run python scripts/ingestion/ingest_fduk.py \
    --start_year 0 --end_year 25 \
    --leagues 'E0' 'I1' \
    --columns 'home_team' 'away_team' 'result' 'odds_h_b365' \
    --save_filename 'ingested_fduk_data' \
    --enrich False
```

### MLflow Server
```bash
# Start MLflow tracking server (runs on localhost:5001)
bash scripts/mlflow_server/mlflow_server_setup.sh
```

### Experiments
```bash
# Run experiment with default parameters
bash scripts/experimentation/run_experiment.sh

# Or run directly with custom parameters
poetry run python scripts/experimentation/experiment.py \
    --experiment_name 'my-experiment' \
    --run_id 'unique-id' \
    --training_data_path 'data/train_data.csv' \
    --validation_data_path 'data/val_data.csv' \
    --hidden_units 8 8 \
    --learning_rate 0.001 \
    --num_epochs 1000 \
    --num_samples 500 \
    --league_tag 'epl'
```

## Architecture Overview

### Core Components

**Bayesian Neural Network Pipeline:**
- `src/bayesian_nn.py`: Core BNN implementation using TensorFlow Probability layers
- `src/priors_posteriors.py`: Prior and posterior distribution definitions (currently standard normal)
- `src/experimentation.py`: Experiment orchestration with MLflow integration

**Data Processing Pipeline:**
- `src/ingestion.py`: Data loading from football-data.co.uk and Sportmonks APIs with column mapping
- `src/preprocess.py`: Feature engineering - aggregates team statistics from first half of seasons to predict second half matches
- `src/helper_functions.py`: Utilities including train/test splitting by season and progress callbacks

### Data Flow

1. **Raw Data**: Historical match data ingested from external APIs
2. **Column Mapping**: `config/config.json` standardizes column names across data sources
3. **Feature Engineering**: Team performance statistics aggregated from first half of seasons
4. **Model Training**: Bayesian NN trained on aggregated features to predict match outcomes
5. **Inference**: Probabilistic predictions with uncertainty quantification via sampling
6. **Experiment Tracking**: All runs logged to MLflow with metrics and parameters

### Key Design Patterns

**Season-Based Data Splitting**: Train/test splits preserve temporal structure by splitting on seasons rather than individual matches, preventing data leakage.

**Bayesian Uncertainty**: Uses variational inference with TensorFlow Probability to learn weight distributions rather than point estimates, enabling uncertainty quantification in predictions.

**Feature Aggregation**: Transforms raw match data into team-level statistics (goals scored/conceded, shots, fouls, etc.) averaged over first half of seasons to predict second half performance.

**MLflow Integration**: All experiments automatically logged with hyperparameters, metrics (ROC-AUC, Brier Score), and model artifacts for reproducible research.

### Notebook Usage

The `notebooks/` directory contains Jupyter notebooks for exploration and demonstration. Key notebook: `0_bayesian_nn_example_usage.ipynb` shows the complete pipeline from data preparation through inference with visualization of probabilistic outputs.

## Configuration

- **Column Mapping**: `config/config.json` maps raw data columns to standardized names
- **MLflow Tracking**: Server runs on localhost:5001 with SQLite backend
- **Data Storage**: Raw data in `data/`, processed datasets created during preprocessing
- **Experiment Scripts**: Template bash scripts in `scripts/` directories for common workflows

Note: The bash scripts in `scripts/` are designed to be edited for specific use cases and should not be committed with local changes.

## Commit Guidelines

Use Conventional Commits format for all commits:
- **Title**: Maximum 50 characters, format: `type(scope): description`
- **Body**: Maximum 72 characters per line
- **Types**: feat, fix, docs, style, refactor, test, chore
- **Example**: `feat(model): add uncertainty quantification to BNN`