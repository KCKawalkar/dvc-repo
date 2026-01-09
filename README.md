# GenAI Text Generation Pipeline

This project implements a reproducible Machine Learning pipeline using DVC and MLflow. It fine-tunes a language model and generates text using a Retrieval-Augmented Generation (RAG) approach.

## Pipeline Stages

1.  **Preprocess**: Cleans raw prompts from `data/raw/prompts.csv`.
2.  **Train**: Fine-tunes a `distilgpt2` model on the processed data. Logs metrics to MLflow.
3.  **Generate**: Uses the trained model and a knowledge base (`data/raw/knowledge_base.csv`) to generate answers with RAG and safety checks.

## Usage

Run the entire pipeline:

```bash
dvc repro
```

## Modifying Data

To add new prompts and trigger a pipeline update:

```bash
echo "Explain large language models." >> data/raw/prompts.csv
dvc repro
```

## Configuration

Parameters for training and generation are defined in `params.yaml`.

## Tracking

- **DVC**: Tracks data versions and pipeline dependencies.
- **MLflow**: Tracks experiments, parameters, metrics (loss, safety_score), and artifacts.
