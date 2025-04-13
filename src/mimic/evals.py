"""Accurate per-topk evaluation with directory structure validation"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import torch
from absl import app, flags
from transformers import T5ForConditionalGeneration, T5Tokenizer

FLAGS = flags.FLAGS
flags.DEFINE_string("predictions_dir", "", "Path to attribution output directory")
flags.DEFINE_string("results_dir", "results", "Output directory for evaluation results")
flags.DEFINE_string("device", "cuda", "Device for inference")

# Load model once
tokenizer = T5Tokenizer.from_pretrained("google/t5_xxl_true_nli_mixture")
model = T5ForConditionalGeneration.from_pretrained(
    "google/t5_xxl_true_nli_mixture",
    device_map="auto",
).eval()

def process_single_file(file_path: Path, topk: int) -> tuple[int, int]:
    """Process one attribution file, return (supported, total)"""
    try:
        with open(file_path, "r") as f:
            queries = json.load(f)
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return 0, 0

    supported = 0
    total = 0
    
    for query in queries:
        total += 1
        passages = query.get("retrieved_paragraphs", [])[:topk]
        if not passages:
            continue
            
        q_text = query["query"]
        question, answer = (q_text.split("?", 1) + ["", ""])[:2]
        question = f"{question.strip()}?" if "?" in q_text else q_text.strip()
        answer = answer.strip()

        # Batch process passages for efficiency
        inputs = [
            f"premise: {p} hypothesis: The answer to '{question}' is '{answer}'"
            for p in passages
        ]
        batch = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(FLAGS.device)
        
        with torch.no_grad():
            outputs = model.generate(**batch)
        
        checks = [tokenizer.decode(o, skip_special_tokens=True) == "1" for o in outputs]
        supported += any(checks)

    return supported, total

def evaluate_method(method_dir: Path, results_dir: Path) -> Dict[str, float]:
    """Evaluate all top-k directories for one method"""
    method_name = method_dir.name
    print(f"\nEvaluating method: {method_name}")
    
    results = {}
    
    # Process each top-k directory
    for topk_dir in sorted(method_dir.glob("top*")):
        if not topk_dir.is_dir():
            continue
            
        try:
            topk = int(topk_dir.name[3:])  # Extract number from 'top1', 'top2' etc.
        except ValueError:
            continue
            
        print(f"  Processing top-{topk}")
        
        # Find all JSON files recursively
        files = list(topk_dir.rglob("*.json"))
        if not files:
            print(f"    No JSON files found in {topk_dir}")
            continue
            
        # Create output directory
        output_dir = results_dir / method_name / f"top{topk}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_supported = 0
        total_questions = 0
        
        # Process files with progress bar
        for file_path in tqdm(files, desc=f"    {method_name} top-{topk}"):
            supported, total = process_single_file(file_path, topk)
            total_supported += supported
            total_questions += total
            
            # Save individual file results
            result_path = output_dir / file_path.relative_to(topk_dir)
            result_path.parent.mkdir(parents=True, exist_ok=True)  # Added parents=True
            with open(result_path, "w") as f:
                json.dump({
                    "file": str(file_path),
                    "supported": supported,
                    "total": total
                }, f)
        
        # Calculate and store scores
        if total_questions > 0:
            score = total_supported / total_questions
            results[str(topk)] = {
                "score": round(score, 4),
                "supported": total_supported,
                "total": total_questions
            }
        else:
            results[str(topk)] = {"score": 0.0, "supported": 0, "total": 0}

        print("#"*20)
        print(" Results for top-{topk}:")
        print(results[str(topk)])
        print("#"*20)

    return results

def main(_):
    # Validate directory structure
    pred_root = Path(FLAGS.predictions_dir)
    if not pred_root.exists():
        raise ValueError(f"Predictions directory not found: {pred_root}")
    
    # Create results directory
    results_root = Path(FLAGS.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    
    final_report = {}
    
    # Find and process each method directory
    for method_dir in pred_root.iterdir():
        if method_dir.is_dir() and method_dir.name in {"bm25", "gtr"} or method_dir.name.startswith("tss"):
            method_results = evaluate_method(method_dir, results_root)
            final_report[method_dir.name] = method_results
            
    
    # Save final report
    report_path = results_root / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    
    # Print summary
    print("\nFinal Evaluation Results:")
    for method, topks in final_report.items():
        print(f"\nMethod: {method}")
        for topk, metrics in sorted(topks.items(), key=lambda x: int(x[0])):
            print(f"  Top-{topk}:")
            print(f"    Score: {metrics['score']:.4f}")
            print(f"    Supported: {metrics['supported']}/{metrics['total']}")

if __name__ == "__main__":
    app.run(main)