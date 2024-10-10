import torch
from src.alphazero.node import Node
from src.utils.game_context import GameContext


def generate_probabilty_target(root_node: Node, context: GameContext) -> torch.Tensor:
    """
    Generates a probability target tensor, which is used to train the neural network.
    The probability target tensor is a tensor containing the probability values for each action in the current state.

    Parameters:
    - node: Node - The node containing the probability values for the current state
    - context: GameContext - The game context object, contains information on how many distinct actions there are in the game

    Returns:
    - torch.Tensor - The probability target tensor
    """
    normalized_root_node_children_visits = torch.zeros(context.num_actions, device=context.device)

    parent_visits = root_node.visits - 1 # Root node gets one more visit than its children (due to the very first selection step)
    for child in root_node.children:
        normalized_root_node_children_visits[child.action] = child.visits / parent_visits
    return normalized_root_node_children_visits