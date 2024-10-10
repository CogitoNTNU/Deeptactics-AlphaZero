import torch
from src.neuralnet.neural_network import NeuralNetwork

state_tensor_buffer = {}
policy_value_buffer = {}

def get_state_tensor(observation_tensor: list[int], shape: list[int], device: torch.device) -> torch.Tensor:
    """
    Get the state tensor of the input node.
    If the state tensor is already calculated, return it from the buffer.
    Otherwise, calculate the state tensor and store it in the buffer.

    Parameters:
    - state: Node - The node to get the state tensor from
    - context: GameContext - Information about the shape of the state tensor and device.

    Returns:
    - torch.Tensor - The state tensor of the input node
    """
    observation_key = tuple(observation_tensor)
    if observation_key in state_tensor_buffer:
        return state_tensor_buffer[observation_key]
    else:
        state_tensor = torch.tensor(observation_key, device=device).reshape(shape)
        state_tensor_buffer[observation_key] = state_tensor
        return state_tensor

def evaluate(observation_tensor: list[int], shape: list[int], nn: NeuralNetwork, device: torch.device) -> tuple[torch.Tensor, float]:
    """
    Neural network evaluation of the state of the input node.
    Will not be run on a leaf node (terminal state)

    
    Forward propagates the state tensor through the neural network.
    Does some reshaping behind the scenes to make the state tensor compatible with the neural network.

    Parameters:
    - state: torch.Tensor - The state tensor to forward propagate
    - context: GameContext - Information about the shape of the state tensor, neural network and device.

    Returns:
    - torch.Tensor - The output of the neural network after forward propagating the state tensor
    
    """
    observation_key = tuple(observation_tensor)
    if observation_key in policy_value_buffer:
        return policy_value_buffer[observation_key]
    else:
        state_tensor = get_state_tensor(observation_tensor, shape, device)
        with torch.no_grad(): ## Disable gradient calculation
            policy, value = nn.forward_for_alphazero(state_tensor)
        policy_value_buffer[observation_key] = (policy, value)
        return policy, value
   