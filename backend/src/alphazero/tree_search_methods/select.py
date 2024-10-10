import torch
from src.alphazero.node import Node


def vectorized_select(node: Node, c: float) -> Node:
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
        kids_visits = node.children_visits
        
        # Compute PUCT for all children in a vectorized manner
        PUCT = torch.where(kids_visits > 0, node.children_values / kids_visits, kids_visits).add_(node.children_policy_values.div(kids_visits.add(torch.ones_like(kids_visits))).mul(const))

        node = node.children[torch.argmax(PUCT).item()] # Return the best child node based on PUCT value
    
    return node