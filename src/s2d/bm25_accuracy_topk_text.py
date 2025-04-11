import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import argparse
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description="BM25 Top-K")
parser.add_argument("--k", type=int, default=3, help="Top-k value for retrieval")
parser.add_argument("--data", type=int, required=True, description="Dataset path")
parser.add_argument("--guideq", type=int, required=True, description="GuideQ dataset path")
args = parser.parse_args()


if not args.data.endswith('.csv'):
    raise ValueError("The data file must be a CSV file.")

if not args.guideq.endswith('.csv'):
    raise ValueError("The guideq file must be a CSV file.")


def bm25_retrieval(k=3):
    # Load data
    original_df = pd.read_csv(args.data)
    original_df['label'] = original_df['label'].str.strip().fillna('MISSING_LABEL')
    
    new_df = pd.read_csv(args.guideq,
                         dtype={'generated_question': str},
                         on_bad_lines='warn').reset_index(drop=True)
    
    new_df['generated_question'] = new_df['generated_question'].fillna('')
    new_df['label'] = new_df['label'].str.strip().fillna('MISSING_LABEL')

    corpus = original_df['text'].tolist()
    labels = original_df['label'].tolist()
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Add attribution columns
    for i in range(k):
        new_df[f'attribution_text_{i+1}'] = ''
        new_df[f'attribution_label_{i+1}'] = ''
        new_df[f'attribution_score_{i+1}'] = 0.0

    label_match_count = 0
    text_match_count = 0
    total = len(new_df)

    for idx, row in tqdm(new_df.iterrows(), total=total, desc="Processing queries"):
        query = row['generated_question']
        true_text = row['text'].strip()
        true_label = row['label'].strip()

        tokenized_query = word_tokenize(query.lower())
        scores = bm25.get_scores(tokenized_query)
        top_k_indices = scores.argsort()[-k:][::-1]

        found_label_match = False
        found_text_match = False

        for i, index_val in enumerate(top_k_indices):
            retrieved_text = corpus[index_val]
            retrieved_label = labels[index_val]

            new_df.at[idx, f'attribution_text_{i+1}'] = retrieved_text
            new_df.at[idx, f'attribution_label_{i+1}'] = retrieved_label
            new_df.at[idx, f'attribution_score_{i+1}'] = scores[index_val]

            if retrieved_label == true_label:
                found_label_match = True
            if retrieved_text.strip() == true_text:
                found_text_match = True

        if found_label_match:
            label_match_count += 1
        if found_text_match:
            text_match_count += 1

    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = f"results/top_{k}_s2d_text_match_bm25.csv"
    new_df.to_csv(output_path, index=False)

    # Print metrics
    print(f"\nRetrieval Metrics (Top-{k}):")
    print(f"Label Match Accuracy: {label_match_count / total:.2%}")
    print(f"Exact Text Match Accuracy: {text_match_count / total:.2%}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    bm25_retrieval(k=args.k)
