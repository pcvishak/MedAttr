import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from utils import extract_passages  # Assuming extract_passages is in utils.py

def create_index_pipeline(input_root: str, output_root: str):
    """
    Creates FAISS indexes for all JSON files in directory structure
    
    Args:
        input_root: Root directory containing EHR JSON files
        output_root: Directory to save indexes with same structure
    """
    # Initialize embedding model
    model = SentenceTransformer('sentence-transformers/gtr-t5-base')
       
    # Process directory structure
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith('.json'):
                # Create matching output structure
                rel_path = os.path.relpath(root, input_root)
                output_dir = os.path.join(output_root, rel_path)
                os.makedirs(output_dir, exist_ok=True)
                
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, f"{"".join(file.split(".")[:-1])}.index")
                
                # Skip existing indexes
                if os.path.exists(output_path):
                    print(f"Skipping existing index: {output_path}")
                    continue
                               
                    # Load and parse EHR data
                with open(input_path, 'r') as f:
                    data = json.load(f)
                
                # Extract passages
                passages = extract_passages(data)
                if not passages:
                    print(f"No passages found in {input_path}")
                    continue
                
                # Generate embeddings
                embeddings = model.encode(passages, convert_to_numpy=True)

                # Normalize for Inner Product (cosine similarity)
                faiss.normalize_L2(embeddings)
                
                dimension = embeddings.shape[1]
                # Create and save index
                index = faiss.IndexFlatIP(dimension)
                index.add(embeddings)
                faiss.write_index(index, output_path)
                
                print(f"Created index: {output_path} ({len(passages)} passages)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, 
                       help="Root directory of PDD JSON files")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for FAISS indexes")
    args = parser.parse_args()    
    create_index_pipeline(args.input, args.output)