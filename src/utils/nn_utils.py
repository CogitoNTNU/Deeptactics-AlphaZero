import torch
import pyspiel
from src.neuralnet.neural_network import NeuralNetwork

def forward_state(state: torch.Tensor, shape: list[int], device: torch.device, neural_network: NeuralNetwork) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward propagates the state tensor through the neural network.
    Does some reshaping behind the scenes to make the state tensor compatible with the neural network.

    Parameters:
    - state: torch.Tensor - The state tensor to forward propagate
    - neural_network - The neural network to forward propagate the state tensor through

    Returns:
    - torch.Tensor - The output of the neural network after forward propagating the state tensor
    """

    state_tensor = torch.reshape(torch.tensor(state.observation_tensor(), device=device), shape).unsqueeze(0) ## Reshape the state tensor to the correct shape and add a batch dimension
    policy, value = neural_network.forward(state_tensor) ## Forward propagate the state tensor through the neural network
    del state_tensor ## Delete the state tensor to free up memory
    return policy.squeeze(0), value.squeeze(0) ## Remove the batch dimension from the output tensors and return them

def reshape_pyspiel_state(state: pyspiel.State, shape: list[int], device: torch.device) -> torch.Tensor:
    """
    Reshapes the pyspiel state tensor to the correct shape for the neural network.

    Parameters:
    - state: pyspiel.State - The state to reshape
    - shape: list[int] - The shape to reshape the state tensor to
    - device: torch.device - The device to place the tensor on, usually 'cuda' or 'cpu'

    Returns:
    - torch.Tensor - The reshaped state tensor
    """

    return torch.tensor(state.observation_tensor(), dtype=torch.float, device = device).reshape(shape).unsqueeze(0) # Reshape the state tensor and add a batch dimension