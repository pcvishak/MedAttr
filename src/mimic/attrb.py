import os
import json
import subprocess
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import faiss
import torch
from typing import List, Dict
import faiss
from utils import extract_passages

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttributionModule:
    def __init__(
        self,
        index: faiss.IndexFlat,
        passages:List[str],
        model_name:str = "sentence-transformers/gtr-t5-base",
        device=device,
        
    ):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)

        if index is None:
            self.index = None
        else:
            self.index = index
        self.passages = passages

    def retrieve_paragraphs(self, texts, k=1, use_bm25  = False):
        results = []
        if (not use_bm25) and (self.index is not None):
            query_embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=16,
                convert_to_numpy=True,
                device=self.device,
            )
            D, I = self.index.search(query_embeddings, k)

            
            for i, query in enumerate(tqdm(texts, desc="Processing query results")):
                retrieved_paragraphs = [self.passages[idx] for idx in I[i]]
                results.append(
                    {   
                        "query_id": i,
                        "query": query,
                        "retrieved_paragraphs": retrieved_paragraphs,
                        "distances": D[i].tolist(),
                    }
                )
        
        else:
            from rank_bm25 import BM25Okapi
            from nltk.tokenize import word_tokenize

            tokenized_corpus = [word_tokenize(p.lower()) for p in self.passages]
            bm25 = BM25Okapi(tokenized_corpus)
            for i, query in enumerate(tqdm(texts, desc="Processing query results")):
                tokenized_query = word_tokenize(query.lower())
                scores = bm25.get_scores(tokenized_query)
                top_k_indices = np.argsort(scores)[-k:][::-1]
                retrieved_paragraphs = [self.passages[idx] for idx in top_k_indices]
                results.append(
                    {   
                        "query_id": i,
                        "query": query,
                        "retrieved_paragraphs": retrieved_paragraphs,
                        "distances": [scores[idx] for idx in top_k_indices],
                    }
                )
        
        return results

    def selection_tss(self, texts, retrieval_results, budget, alpha=0.6, submodular_function="fl"):
        from submodlib import FacilityLocationMutualInformationFunction, GraphCutMutualInformationFunction
        from bias_selector import BiasedSubsetSelector
        
        submod_map = {
            "fl": FacilityLocationMutualInformationFunction,
            "gc": GraphCutMutualInformationFunction,
        }

        selected_results = []

        for result in tqdm(retrieval_results, desc="Processing queries"):
            # Get query context
            query_id = result["query_id"]
            query = result["query"]
            retrieved_passages = result["retrieved_paragraphs"]
            distances = result["distances"]
            
            # Get passage embeddings for retrieved passages
            passage_indices = [self.passages.index(p) for p in retrieved_passages]
            passage_embeds = self.index.reconstruct_batch(passage_indices)
            
            # Split question and answer
            if "?" in query:
                question, answer = query.split("?", 1)
                question = question.strip() + "?"
                answer = answer.strip()
            else:
                question, answer = query, ""
            
            # Encode the answer as the target query for subset selection
            answer_embed = self.model.encode([answer], show_progress_bar=False, device=self.device)
            
            # Create the submodular selection function based on the answer embedding
            obj = submod_map[submodular_function](
                n=len(retrieved_passages),
                num_queries=1, 
                data=passage_embeds,  
                queryData=answer_embed, 
                metric="cosine"
            )
            
            # Create biased subset selector
            selector = BiasedSubsetSelector(
                data=passage_embeds,
                retriever_ranks=list(range(len(retrieved_passages))),
                retriever_scores=distances,
                submodular_function=obj,
                alpha=alpha,  # You can adjust alpha as needed
            )
            
            # Get selected indices
            selected_indices = selector.select(budget)
            
            # Update results with selected passages
            selected_results.append({
                "query_id": query_id,
                "query": query,
                "retrieved_paragraphs": [retrieved_passages[i] for i in selected_indices],
                "distances": [distances[i] for i in selected_indices]
            })
        
        return selected_results


    def get_attribution_tss(self, texts:List[str], budget:int, retriever_budget=30, alpha=0.6, submodular_function="flmi"):
        retrieval_results = self.retrieve_paragraphs(texts, k=min(retriever_budget, len(self.passages)))
        selected_results = self.selection_tss(texts, retrieval_results, budget, alpha=alpha, submodular_function=submodular_function)
        return selected_results