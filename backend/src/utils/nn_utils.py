import torch
import pyspiel
from src.utils.game_context import GameContext

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