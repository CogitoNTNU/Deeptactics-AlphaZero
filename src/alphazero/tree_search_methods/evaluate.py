import torch
from src.alphazero.node import Node
from src.utils.game_context import GameContext
from src.utils.nn_utils import forward_state


def evaluate(node: Node, context: GameContext) -> tuple[torch.Tensor, float]:
    """
    Neural network evaluation of the state of the input node.
    Will not be run on a leaf node (terminal state)
    """
    policy, value = forward_state(node.state, context)
    return policy, value.item()