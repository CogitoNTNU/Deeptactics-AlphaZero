import torch
import pyspiel
from src.neuralnet.neural_network import NeuralNetwork

class GameContext:
    
    def __init__(self, game_name: str, nn: NeuralNetwork, save_path: str):
        
        self.game = pyspiel.load_game(game_name)
        """
        The pyspiel game object.
        """

        self.num_actions = self.game.num_distinct_actions()
        """
        The number of distinct actions in the game.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """
        Device used to perform matmul operations. If cuda is available, GPU will be used.
        """

        self.nn = nn.to(self.device)
        """
        The neural network assigned to this game context.
        """

        self.save_path = save_path
        """
        The path to save the neural network model. If None, the model will not be saved.
        """
        
    def set_neural_network(self, nn: NeuralNetwork) -> None:
        """
        Method for changing the neural network assigned to this game context.
        """
        self.nn = nn.to(self.device)
    
    def set_save_path(self, save_path: str) -> None:
        """
        Method for changing the save path of the neural network model.
        """
        self.set_save_path(save_path)    

    def get_initial_state(self) -> pyspiel.State:
        """
        Get a fresh initial state of the game assigned to this context.
        """
        return self.game.new_initial_state()

    