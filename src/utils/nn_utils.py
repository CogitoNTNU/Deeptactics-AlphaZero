import torch
import pyspiel
from src.utils.game_context import GameContext

def forward_state(state: torch.Tensor, context: GameContext) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward propagates the state tensor through the neural network.
    Does some reshaping behind the scenes to make the state tensor compatible with the neural network.

    Parameters:
    - state: torch.Tensor - The state tensor to forward propagate
    - context: GameContext - Information about the shape of the state tensor, neural network and device.

    Returns:
    - torch.Tensor - The output of the neural network after forward propagating the state tensor
    """
    shape = context.game.observation_tensor_shape() ## Get the shape of the state tensor
    state_tensor = torch.reshape(torch.tensor(state.observation_tensor(), device=context.device), shape).unsqueeze(0) ## Reshape the state tensor to the correct shape and add a batch dimension
    
    with torch.no_grad(): ## Disable gradient calculation
        policy, value = context.nn.forward(state_tensor) ## Forward propagate the state tensor through the neural network
    del state_tensor ## Delete the state tensor to free up memory
    
    return policy.squeeze(0), value.squeeze(0) ## Remove the batch dimension from the output tensors and return them

def reshape_pyspiel_state(state: pyspiel.State, context: GameContext) -> torch.Tensor:
    """
    Reshapes the pyspiel state tensor to the correct shape for the neural network.

    Parameters:
    - state: pyspiel.State - The state to reshape
    - context: GameContext - Information about the shape of the state tensor and device.

    Returns:
    - torch.Tensor - The reshaped state tensor
    """
    
    shape = context.game.observation_tensor_shape() ## Get the shape of the state tensor
    return torch.tensor(state.observation_tensor(), dtype=torch.float, device = context.device).reshape(shape).unsqueeze(0) # Reshape the state tensor and add a batch dimension