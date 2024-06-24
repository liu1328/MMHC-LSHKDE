from itertools import combinations

def subsets(nbrs, k):
    """
       Generate all subsets of size k from the set of neighbors nbrs.
       Args:
           nbrs (set): A set of neighbors.
           k (int): The size of the subsets to generate.
       Returns:
           set: A set of subsets, each of size k.
       """
    return set(combinations(nbrs, k))

