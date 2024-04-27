import torch
from src.alphazero.node import Node

def vectorized_select(node: Node, c: float) -> Node: # OPTIMIZATION for GPU, great speedup is expected when number of children is large.
    """
    Select stage of MCTS.
    Go through the game tree, layer by layer.
    Chooses the node with the highest UCB-score at each layer.
    Returns a leaf node.
    """

    while node.has_children():

        if node.visits == 1: # Special case, saves some computation time
            return node.children[torch.argmax(node.children_policy_values).item()]        
        
        const = c * node.visits**0.5

        # Compute PUCT for all children in a vectorized manner
        Q = torch.where(node.children_visits > 0, node.children_values / node.children_visits, torch.zeros_like(node.children_values))
        Q.add_(const * (node.children_policy_values / node.children_visits.add(torch.ones_like(node.children_visits)) ))

        node = node.children[torch.argmax(Q).item()] # Return the best child node based on PUCT value
    
    return node