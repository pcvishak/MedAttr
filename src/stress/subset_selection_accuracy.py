import pandas as pd
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from submodlib import FacilityLocationMutualInformationFunction, GraphCutMutualInformationFunction, LogDeterminantMutualInformationFunction
from tqdm import tqdm
import os
import argparse
import re

# Import your custom classes
from bias_selector import BiasedSubsetSelector

parser = argparse.ArgumentParser(description="TSS Attribution")
parser.add_argument("--data", type=int, required=True, description="Dataset path")
parser.add_argument("--guideq", type=int, required=True, description="GuideQ dataset path")
parser.add_argument("--k", type=int, default=3, help="Final attribution depth")
parser.add_argument("--submodular_function", type=str, default="GC", choices=["FL", "GC", "LD"], help="Submodular function to use")
parser.add_argument("--alpha", type=float, default=0.6, help="Alpha value for submodular selection")
args = parser.parse_args()


def tss_retrieval(k=3):
    # Load data
    original_df = pd.read_csv(args.data)
    new_df = pd.read_csv(args.guideq, dtype={'generated_question': str}, on_bad_lines='warn').reset_index(drop=True)
    
    # Clean data
    new_df['generated_question'] = new_df['generated_question'].fillna('')
    new_df['label'] = new_df['label'].str.strip().fillna('MISSING_LABEL')
    
    # Load models
    encoder = SentenceTransformer('sentence-transformers/gtr-t5-base').half()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load FAISS index
    index = faiss.read_index('symptom2disease_index.faiss')

    # Add attribution columns
    for i in range(k):
        new_df[f'attribution_text_{i+1}'] = ''
        new_df[f'attribution_label_{i+1}'] = ''
        new_df[f'attribution_score_{i+1}'] = 0.0

    label_match_count = 0
    text_match_count = 0
    total = len(new_df)
    INITIAL_RETRIEVAL = 30

    for idx, row in tqdm(new_df.iterrows(), total=total, desc="Processing queries"):
        query = row['generated_question']
        true_label = row['label']
        original_text = row['text']
        
        # Stage 1: Initial FAISS retrieval
        query_embed = encoder.encode([query])
        scores, indices = index.search(query_embed, INITIAL_RETRIEVAL)
        
        # Collect candidates
        candidates = []
        candidate_embeddings = []
        for score, index_val in zip(scores[0], indices[0]):
            if index_val >= 0:
                text = original_df.iloc[index_val]['text']
                label = original_df.iloc[index_val]['subreddit']
                candidates.append({
                    'text': text,
                    'label': label,
                    'faiss_score': score,
                    'index': index_val
                })
                candidate_embeddings.append(encoder.encode(text))
        
        if len(candidates) == 0:
            continue

        top3_predicted_labels =  re.findall(r"'([^']*)'", row['top3_predicted_labels'])  # Assuming this is already in the dataframe

        top3_label_embeddings = encoder.encode(top3_predicted_labels, convert_to_numpy=True)

        retriever_ranks = list(range(len(candidates)))  

        if args.submodular_function == "FL":
            sub = FacilityLocationMutualInformationFunction(
                    n=len(candidates),
                    num_queries=len(top3_label_embeddings), 
                    data=np.array(candidate_embeddings),
                    queryData=top3_label_embeddings,  
                )
        elif args.submodular_function == "GC":
            sub = GraphCutMutualInformationFunction(
                n=len(candidates),
                num_queries=len(top3_label_embeddings),  
                data=np.array(candidate_embeddings),
                queryData=top3_label_embeddings,  
            )
        elif args.submodular_function == "LD":
            sub = LogDeterminantMutualInformationFunction(
                lambdaVal=1,
                n=len(candidates),
                num_queries=len(top3_label_embeddings),
                data=np.array(candidate_embeddings),
                queryData=top3_label_embeddings,  
            )

        submod_selector = BiasedSubsetSelector(
            data=np.array(candidate_embeddings),
            retriever_ranks=retriever_ranks,
            submodular_function=sub,
            retriever_scores=np.array([c['faiss_score'] for c in candidates]),
            alpha=args.alpha,
        )
        
        selected_indices = submod_selector.select(k)
        final_candidates = [candidates[i] for i in selected_indices]
        
        # Store results
        found_label = False
        found_text = False
        for i, cand in enumerate(final_candidates[:k]):
            new_df.at[idx, f'attribution_text_{i+1}'] = cand['text']
            new_df.at[idx, f'attribution_label_{i+1}'] = cand['label']
            new_df.at[idx, f'attribution_score_{i+1}'] = cand['faiss_score']
            
            if cand['label'] == true_label:
                found_label = True
            if cand['text'] == original_text:
                found_text = True

        # Update counts
        label_match_count += int(found_label)
        text_match_count += int(found_text)

    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = f"results/s2d_tss_top_{k}_alpha_{args.alpha}.csv"
    new_df.to_csv(output_path, index=False)
    
    # Print metrics
    print(f"\nRetrieval Metrics (Targeted Submod Top-{k}):")
    print(f"Label Match Accuracy: {label_match_count/total:.2%}")
    print(f"Exact Text Match Accuracy: {text_match_count/total:.2%}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    tss_retrieval(k=args.k)
