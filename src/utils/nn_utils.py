import torch
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
    return policy.squeeze(0), value.squeeze(0) ## Remove the batch dimension from the output tensors and return them
