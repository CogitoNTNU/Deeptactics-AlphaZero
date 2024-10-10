import torch.multiprocessing as mp
from src.alphazero.agents.alphazero_training_agent import AlphaZero
from src.neuralnet.neural_network import NeuralNetwork

def get_play_alphazero_games_arguments(
        alphazero: AlphaZero, num_games: int, num_simulations: int
        ) -> tuple[list[tuple[AlphaZero, NeuralNetwork, int, int]], int]:
    """
    Parameters:
    - alphazero: AlphaZero - The AlphaZero training agent
    - nn: NeuralNetwork - The neural network used to predict the probabilities and values
    - num_games: int - The number of games to play
    - num_simulations: int - The number of simulations to run for each move

    Returns:
    - arguments: list[tuple[AlphaZero, NeuralNetwork, int, int]] - Returns a list of tuples containing the arguments for the play_alphazero_game function.
    - number_of_threads: int - The number of threads to use for multiprocessing
    """
    
    max_num_threads = mp.cpu_count() - 1
    number_of_threads = max(1, min(max_num_threads, num_games // 10)) # We estimate that we should have at least 4 games per process to get the best time efficiency.
    
    num_games_per_thread = num_games // number_of_threads
    remainder = num_games % number_of_threads

    arguments = [(alphazero, num_games_per_thread, num_simulations) for _ in range(number_of_threads - remainder)]
    arguments.extend([(alphazero, num_games_per_thread + 1, num_simulations) for _ in range(remainder)])
    
    return arguments, number_of_threads