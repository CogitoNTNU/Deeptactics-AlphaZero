import torch
from torch.testing import assert_close
from src.utils.random_utils import generate_dirichlet_noise
from src.utils.game_context import GameContext
from src.neuralnet.neural_network import NeuralNetwork

def test_generate_dirichlet_noise():
    """
    Tests the generate_dirichlet_noise function to ensure it generates a tensor
    of the correct shape and dtype.
    """
    alpha = 0.3
    context = GameContext("connect_four", NeuralNetwork(), None)
    context.device = torch.device("cpu")
    num_legal_actions = 4

    # Assuming alpha = 0.3 and num_actions = 4
    # Manual seed 75 -> expected_noise = tensor([0.3911, 0.5310, 0.0039, 0.0740])
    expected_noise = torch.tensor([0.3911, 0.5310, 0.0039, 0.0740], dtype=torch.float, device=context.device)

    torch.manual_seed(75)
    actual_noise = generate_dirichlet_noise(context, num_legal_actions, alpha)
    assert_close(actual_noise, expected_noise, atol=1e-4, rtol=0)
