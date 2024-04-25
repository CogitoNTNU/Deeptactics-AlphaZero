from src.alphazero.alphazero_training_agent import AlphaZero
from src.neuralnet.neural_network import NeuralNetwork
from src.utils.multi_core_utils import get_play_alphazero_games_arguments
import torch.multiprocessing as mp

"""
In these tests, we are assuming that the minimum amount of games to play
per thread is 4. It takes time to set up the threads, so we want to make
sure that we are playing enough games to make the overhead worth it.
"""

def test_argument_generation():

    alphazero = AlphaZero(game_name="tic_tac_toe")
    nn = NeuralNetwork().to(alphazero.device)
    num_simulations = 1000

    def test_threads(expected_threads, num_games):
        arguments, number_of_threads = get_play_alphazero_games_arguments(
            alphazero, nn, num_games, num_simulations
        )
        assert len(arguments) == number_of_threads
        assert number_of_threads == expected_threads
        assert sum([args[2] for args in arguments]) == num_games

    def test_two_threads(num_games: int):
        test_threads(2, num_games)

    test_two_threads(8)
    test_two_threads(9)
    test_two_threads(10)
    test_two_threads(11)
    test_threads(3, 12)

def test_max_threads():
    max_threads = mp.cpu_count()
    alphazero = AlphaZero(game_name="tic_tac_toe")
    nn = NeuralNetwork().to(alphazero.device)
    num_simulations = 1000

    def test_threads(expected_threads, num_games):
        arguments, number_of_threads = get_play_alphazero_games_arguments(
            alphazero, nn, num_games, num_simulations
        )
        assert len(arguments) == number_of_threads
        assert number_of_threads == expected_threads
        assert sum([args[2] for args in arguments]) == num_games
    
    test_threads(max_threads, 5*max_threads)
    test_threads(max_threads, 5*max_threads + 3)



    
