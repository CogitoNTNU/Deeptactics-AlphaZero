import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, hidden_dim: int, game_shape: tuple[int, int]):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LayerNorm([hidden_dim, game_shape[0], game_shape[1]]),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LayerNorm([hidden_dim, game_shape[0], game_shape[1]]),
        )

    def forward(self, x):
        residual = x
        x = self.conv_block(x)
        x += residual
        x = F.relu(x)
        return x
    
class NeuralNetworkConnectFour(nn.Module):
    def __init__(
        self,
        hidden_dimension=256,
        input_dimension=3,
        res_blocks=5,
        game_shape=(6, 7),
        game_size=42,
        legal_moves=7,
    ):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.input_dimension = input_dimension
        self.res_blocks = res_blocks

        self.initial = nn.Sequential(
            nn.Conv2d(
                self.input_dimension,
                out_channels=self.hidden_dimension,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # Convolution matrix
            nn.LayerNorm([self.hidden_dimension, game_shape[0], game_shape[1]]),  # Layer normalization
            nn.ReLU(),  # Activation function
        )
        
        self.residual_blocks = nn.ModuleList(
            [ResBlock(hidden_dimension, game_shape) for _ in range(self.res_blocks)]
        )

        self.policy = nn.Sequential(
            nn.Conv2d(self.hidden_dimension, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game_size, legal_moves)
        )

        self.value = nn.Sequential(
            nn.Conv2d(self.hidden_dimension, out_channels=1, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LayerNorm(1 * game_size),
            nn.Linear(1 * game_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.initial(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value
    
    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        
        # Check if the directory exists
        if not os.path.exists(directory):
            # If it doesn't exist, create it
            os.makedirs(directory)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls, path: str, hidden_dimension=256, input_dimension=3, res_blocks=5, game_shape=(6, 7), game_size=42, legal_moves=7
        ):
        """
        Loads the model from the specified path.

        Parameters:
        - path: str - The path to the saved model state dictionary.
        - hidden_dimension: int - The dimension of the hidden layers.
        - input_dimension: int - The input dimension.
        - res_blocks: int - The number of residual blocks.
        - game_size: int - The size of the game board.
        - legal_moves: int - The number of legal moves.

        Returns:
        - An instance of NeuralNetwork with weights loaded from the specified path.
        """
        
        model = cls(
            hidden_dimension=hidden_dimension, 
            input_dimension=input_dimension, 
            res_blocks=res_blocks, 
            game_shape=game_shape,
            game_size=game_size, 
            legal_moves=legal_moves)        
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        
        return model
    
