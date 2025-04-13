import os
import json
import faiss
import argparse
from tqdm.auto import tqdm
from attrb import AttributionModule
from utils import extract_passages

def save_results(method, k, results, qa_path, output_root, alpha=0.6, submodf="fl"):
    """Save results in method-specific directory structure"""
    rel_path = os.path.relpath(qa_path, args.qa_dir)
    base_name = os.path.basename(qa_path)
    
    if method == "tss":
        method_dir = os.path.join(output_root, f"{method}_{submodf}_alpha{alpha}", f"top{k}")
    else:
        method_dir = os.path.join(output_root, method, f"top{k}")
    
    output_path = os.path.join(method_dir, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def get_associated_paths(qa_path, pdd_dir, index_dir):
    """Direct path mapping using directory structure"""
    # Get relative path from QA directory
    rel_path = os.path.relpath(qa_path, args.qa_dir)
    
    # Construct PDD path
    pdd_path = os.path.join(pdd_dir, rel_path)
    
    # Construct index path (replace .json with .index)
    index_path = "".join(os.path.join(index_dir, rel_path).split(".")[:-1] + [".index"])
    
    # Validate paths
    if not os.path.exists(pdd_path):
        raise FileNotFoundError(f"PDD file not found: {pdd_path}")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    
    return pdd_path, index_path

def process_qa_file(qa_path, pdd_dir, index_dir, output_root, alpha, submodf):
    try:
        # Get directly mapped paths
        pdd_path, index_path = get_associated_paths(qa_path, pdd_dir, index_dir)
        
        # Load data
        with open(qa_path) as f:
            qa_data = json.load(f)
        with open(pdd_path) as f:
            pdd_data = json.load(f)
        
        # Create attribution module
        passages = extract_passages(pdd_data)
        index = faiss.read_index(index_path)
        attributor = AttributionModule(index, passages)
        
        # Process queries
        queries = [f"{q['question']} {q['answer']}" for q in qa_data]
        
        # Process all retrieval methods
        for k in range(1, 6):
            # BM25
            bm25_results = attributor.retrieve_paragraphs(queries, k=k, use_bm25=True)
            save_results("bm25", k, bm25_results, qa_path, output_root)
            
            # GTR
            gtr_results = attributor.retrieve_paragraphs(queries, k=k, use_bm25=False)
            save_results("gtr", k, gtr_results, qa_path, output_root)
            
            # TSS
            for _alpha in [0.0, 0.2, 0.4, 0.6, 0.8]:
                tss_results = attributor.get_attribution_tss(queries, budget=k, alpha=_alpha, submodular_function=submodf)
                save_results("tss", k, tss_results, qa_path, output_root, _alpha, submodf)
            
            # tss_results = attributor.get_attribution_tss(queries, budget=k, alpha=alpha, submodular_function=submodf)
            # save_results("tss", k, tss_results, qa_path, output_root, alpha, submodf)

    except Exception as e:
        print(f"Error processing {qa_path}: {str(e)}")

def main(args):
    # Collect all QA files
    qa_files = []
    for root, _, files in os.walk(args.qa_dir):
        qa_files.extend(os.path.join(root, f) for f in files if f.endswith('.json'))
    
    # Process files with progress bar
    for qa_path in tqdm(qa_files, desc="Processing QA files"):
        process_qa_file(
            qa_path=qa_path,
            pdd_dir=args.pdd_dir,
            index_dir=args.index_dir,
            output_root=args.output_dir,
            alpha=args.alpha,
            submodf=args.submodular_function
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured Attribution Pipeline")
    parser.add_argument("--qa_dir", required=True, help="Root directory of QA JSON files")
    parser.add_argument("--pdd_dir", required=True, help="Root directory of PDD JSON files")
    parser.add_argument("--index_dir", required=True, help="Root directory of Faiss indexes")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--alpha", type=float, default=0.6, help="TSS alpha parameter")
    parser.add_argument("--submodular_function", type=str, default="fl", 
                      choices=["fl", "gc"], help="Submodular function type")
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    args.qa_dir = os.path.abspath(args.qa_dir)
    args.pdd_dir = os.path.abspath(args.pdd_dir)
    args.index_dir = os.path.abspath(args.index_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    
    # Validate directory structure
    for d in [args.qa_dir, args.pdd_dir, args.index_dir]:
        if not os.path.isdir(d):
            raise ValueError(f"Directory not found: {d}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)