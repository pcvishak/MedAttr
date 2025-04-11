import numpy as np
from submodlib import SetFunction

class BiasedSubsetSelector:
    def __init__(self, data, retriever_ranks, submodular_function: SetFunction, retriever_scores=None, alpha=0.5, base=0.9):
        """
        Parameters:
        data (np.ndarray): Embeddings/features of items (n x d matrix)
        retriever_ranks (list): List of indices representing retriever's ranking order
        submodular_function (SetFunction): Instantiated submodular function from submodlib
        alpha (float): Bias parameter [0,1] where 1=retriever order, 0=diverse
        base (float): Decay base for retriever scores (0 < base <= 1)
        """
        self.n = len(retriever_ranks)
        self.data = data
        self.alpha = alpha
        self.base = base
        self.submod_func = submodular_function
        self.retriever_ranks = retriever_ranks
        # Create retriever scores using exponential decay
        if retriever_scores is not None:
            self.retriever_scores = retriever_scores
        else:
            self.retriever_scores = np.array([base**rank for rank in range(len(retriever_ranks))])
        
        # Create mapping from original index to retriever's rank position
        self.idx_to_rank = {idx: rank for rank, idx in enumerate(retriever_ranks)}

    def select(self, k):
        """
        Greedy selection of k items balancing retriever bias and diversity
        Returns selected indices in the order of selection
        """
        selected = set()
        order = []
        
        for _ in range(k):
            best_gain = -np.inf
            best_idx = None
            
            # Evaluate all candidates not yet selected
            for candidate in range(self.n):
                if candidate in selected:
                    continue
                
                # Get retriever score component
                rank = self.idx_to_rank[candidate]
                retriever_gain = self.alpha * self.retriever_scores[rank]
                
                # Get diversity component using marginal gain
                diversity_gain = (1 - self.alpha) * self.submod_func.marginalGain(selected, candidate)
                
                total_gain = retriever_gain + diversity_gain
                
                if total_gain > best_gain:
                    best_gain = total_gain
                    best_idx = candidate
            
            # Add best candidate to selection
            if best_idx is not None:
                selected.add(best_idx)
                order.append(best_idx)
                
                # Update memoization for submodular function
                # self.submod_func.updateMemoization(selected, best_idx)
        
        # Return indices in original retriever order
        return [self.retriever_ranks[idx] for idx in order]

# Example usage with Facility Location
if __name__ == "__main__":
    from submodlib import FacilityLocationFunction
    
    # Sample data
    embeddings = np.random.randn(100, 256)
    retriever_ranks = list(range(100))  # [0, 1, 2, ..., 99]
    
    # Instantiate submodular function
    fl = FacilityLocationFunction(
        n=100,
        mode="dense",
        data=embeddings,
        metric="cosine" 
    )
    
    # Create selector
    selector = BiasedSubsetSelector(
        data=embeddings,
        retriever_ranks=retriever_ranks,
        submodular_function=fl,
        alpha=0.5,
        base=0.85
    )
    
    # Select top 10 items
    selected = selector.select(10)
    print("Selected indices (in selection order):", selected)