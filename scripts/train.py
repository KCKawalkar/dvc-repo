import argparse
import mlflow
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--data", type=str, default="data/processed.csv", help="Path to training data")
    parser.add_argument("--model_dir", type=str, default="models/model_v1", help="Output directory for model")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    args = parser.parse_args()

    mlflow.set_experiment("genai-text-generation")

    with mlflow.start_run():
        df = pd.read_csv(args.data)

        mlflow.log_param("model", args.model_name)
        mlflow.log_param("num_samples", len(df))
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

        # Dummy loss for POC
        loss = 1.08
        mlflow.log_metric("loss", loss)

        model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)

        mlflow.log_artifacts(args.model_dir)

if __name__ == "__main__":
    main()
