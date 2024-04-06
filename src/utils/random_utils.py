import torch
from torch.distributions.dirichlet import Dirichlet

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