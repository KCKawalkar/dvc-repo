import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument("--input", type=str, default="data/raw/prompts.csv", help="Path to raw data")
    parser.add_argument("--output", type=str, default="data/processed.csv", help="Path to processed data")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["prompt"] = df["prompt"].str.strip()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
