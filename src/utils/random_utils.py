import torch
from torch.distributions.dirichlet import Dirichlet
from src.alphazero.node import Node

def generate_dirichlet_noise(num_actions: int, alpha: float, device: torch.device) -> torch.Tensor:
    """
    Generates a Dirichlet noise tensor, which is used to encourage exploration in the policy values.
    The Dirichlet distribution is a multivariate generalization of the Beta distribution.

    Parameters:
    - num_actions: int - The number of actions in the current state
    - alpha: float - The concentration parameter of the Dirichlet distribution

    Returns:
    - torch.Tensor - The Dirichlet noise tensor
    """
    return Dirichlet(torch.tensor([alpha] * num_actions, dtype=torch.float, device=device)).sample()


def generate_probabilty_target(root_node: Node, num_actions: int, device: torch.device) -> torch.Tensor:
    """
    Generates a probability target tensor, which is used to train the neural network.
    The probability target tensor is a tensor containing the probability values for each action in the current state.

    Parameters:
    - node: Node - The node containing the probability values for the current state
    - num_actions: int - The number of actions in the current state

    Returns:
    - torch.Tensor - The probability target tensor
    """
    normalized_root_node_children_visits = torch.zeros(num_actions, device=device, dtype=torch.float)

    parent_visits = root_node.visits
    for child in root_node.children:
        normalized_root_node_children_visits[child.action] = child.visits / parent_visits
    return normalized_root_node_children_visits