import argparse
import pandas as pd
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

def retrieve_context(prompt, kb_df):
    """Simple retrieval mechanism based on keyword matching."""
    for _, row in kb_df.iterrows():
        if str(row['topic']).lower() in prompt.lower():
            return row['content']
    return "General knowledge."

def check_safety(text):
    """Basic safety filter for toxicity."""
    unsafe_keywords = ["violence", "hate", "kill", "stupid"]
    for word in unsafe_keywords:
        if word in text.lower():
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description="RAG Generation with Safety Checks")
    parser.add_argument("--model_dir", type=str, help="Path to trained model")
    parser.add_argument("--data", type=str, help="Path to input data")
    parser.add_argument("--knowledge_base", type=str, help="Path to knowledge base")
    parser.add_argument("--output", type=str, help="Path to save predictions")
    parser.add_argument("--max_length", type=int, default=50, help="Max generation length")
    args = parser.parse_args()

    # Load resources
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    
    df = pd.read_csv(args.data)
    kb_df = pd.read_csv(args.knowledge_base)
    
    results = []
    
    mlflow.set_experiment("genai-text-generation")
    with mlflow.start_run(run_name="rag_generation"):
        for _, row in df.iterrows():
            prompt = row['prompt']
            context = retrieve_context(prompt, kb_df)
            
            # RAG Prompt Construction
            full_input = f"Context: {context}\nQuestion: {prompt}\nAnswer:"
            inputs = tokenizer(full_input, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids, 
                    max_length=inputs.input_ids.shape[1] + args.max_length,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text[len(full_input):].strip()
            
            is_safe = check_safety(answer)
            
            results.append({
                "prompt": prompt,
                "context": context,
                "generated_answer": answer,
                "is_safe": is_safe
            })
        
        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        output_df = pd.DataFrame(results)
        output_df.to_csv(args.output, index=False)
        
        # Log metrics and artifacts
        safety_score = output_df['is_safe'].mean()
        mlflow.log_metric("safety_score", safety_score)
        mlflow.log_artifact(args.output)

if __name__ == "__main__":
    main()