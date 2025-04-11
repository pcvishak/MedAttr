import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import argparse
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description="GTR Retrieval")
parser.add_argument("--k", type=int, default=3, help="Top-k value for retrieval")
parser.add_argument("--data", type=int, required=True, description="Dataset path")
parser.add_argument("--guideq", type=int, required=True, description="GuideQ dataset path")
args = parser.parse_args()

if not args.data.endswith('.csv'):
    raise ValueError("Dataset path must be a CSV file.")

if not args.guideq.endswith('.csv'):
    raise ValueError("GuideQ dataset path must be a CSV file.")


def gtr_retrieval(k=3):
    # Load data
    original_df = pd.read_csv(args.data)
    original_df['label'] = original_df['label'].str.strip().fillna('MISSING_LABEL')
    new_df = pd.read_csv("args.guideq", 
                        dtype={'generated_question': str}, 
                        on_bad_lines='warn').reset_index(drop=True)
    
    # Clean data
    new_df['generated_question'] = new_df['generated_question'].fillna('')
    new_df['label'] = new_df['label'].str.strip().fillna('MISSING_LABEL')
    
    # Load model and index
    model = SentenceTransformer('sentence-transformers/gtr-t5-base').half()
    index = faiss.read_index('symptom2disease_index.faiss')

    # Add attribution columns
    for i in range(k):
        new_df[f'attribution_text_{i+1}'] = ''
        new_df[f'attribution_stress_label_{i+1}'] = ''
        new_df[f'attribution_score_{i+1}'] = 0.0

    label_match_count = 0
    text_match_count = 0
    total = len(new_df)

    for idx, row in tqdm(new_df.iterrows(), total=total, desc="Processing queries"):
        query = row['generated_question']
        true_label = row['label']
        original_text = row['text']  # Assuming 'text' contains the original reference text
        
        # Get top-k matches
        query_embedding = model.encode([query])
        scores, indices = index.search(query_embedding, k=k)
        
        # Process results
        found_label_match = False
        found_text_match = False
        retrieved_texts = []

        for i, (score, index_val) in enumerate(zip(scores[0], indices[0])):
            if index_val >= 0:
                # Get retrieved item
                retrieved_text = original_df.iloc[index_val]['text']
                retrieved_label = original_df.iloc[index_val]['subreddit']
                
                # Store in dataframe
                new_df.at[idx, f'attribution_text_{i+1}'] = retrieved_text
                new_df.at[idx, f'attribution_stress_label_{i+1}'] = retrieved_label
                new_df.at[idx, f'attribution_score_{i+1}'] = score
                
                # Check matches
                if retrieved_label == true_label:
                    found_label_match = True
                if retrieved_text == original_text:
                    found_text_match = True

        # Update counts
        if found_label_match:
            label_match_count += 1
        if found_text_match:
            text_match_count += 1

    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = f"results/top_{k}_stress_retrieval.csv"
    new_df.to_csv(output_path, index=False)
    
    # Print metrics
    print(f"\nRetrieval Metrics (Top-{k}):")
    print(f"Label Match Accuracy: {label_match_count/total:.2%}")
    print(f"Exact Text Match Accuracy: {text_match_count/total:.2%}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    gtr_retrieval(k=args.k)