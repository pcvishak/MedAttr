import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import argparse

parser = argparse.ArgumentParser(description="Create FAISS index for medical data")
parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
args = parser.parse_args()


# Load the dataset
df = pd.read_csv(args.data)
texts = df['text'].tolist()

# Load the model with half-precision
model = SentenceTransformer('sentence-transformers/gtr-t5-base').half()

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Convert to numpy array in float32 (FAISS requirement)
embeddings = embeddings.astype('float32')

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save the index
faiss.write_index(index, 'stress_index.faiss')

print(f"FAISS index created with {index.ntotal} entries")