"""
This file instantiates the alphazero_training class,
and generates training data by playing games with the alphazero agent.
"""


import torch

from src.alphazero.alphazero_training_agent import AlphaZero
from src.neuralnet.neural_network import NeuralNetwork
from src.utils.nn_utils import reshape_pyspiel_state


def play_alphazero_game(
    alphazero_mcts: AlphaZero, nn: NeuralNetwork, num_simulations: int
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Plays a game using the AlphaZero training agent, and returns a list of training data.
    The number of training data is equal to the number of moves in the game.
    For each move, the training data is as follows:
    (state, probability_target, reward)
    """

    state = alphazero_mcts.game.new_initial_state(); shape = alphazero_mcts.game.observation_tensor_shape()
    game_data = []; move_number = 1

    while not state.is_terminal():
        # print(state, '\n~~~~~~~~~~~~~~~')
        action, probability_visits = alphazero_mcts.run_simulation(
            state, nn, move_number, num_simulations=num_simulations
        )
        game_data.append(
            (
                reshape_pyspiel_state(state, shape, alphazero_mcts.device),
                probability_visits,
            )
        )
        state.apply_action(action)
        move_number += 1

    # print(state, '\n~~~~~~~~~~~~~~~')
    rewards = state.returns()
    return [
        (
            state,
            probability_visits,
            torch.tensor(
                [rewards[i % 2]], dtype=torch.float, device=alphazero_mcts.device
            ),
        )
        for i, (state, probability_visits) in enumerate(game_data)
    ]

def generate_training_data(nn: NeuralNetwork, num_games: int, num_simulations: int = 100):
    """
    Takes in a neural network, and generates training data by making the neural network play games against itself.
    The amount of training data is equal to:
    - sum of (number of moves in each game)

    Parameters:
    - nn: NeuralNetwork - The neural network used to generate training data
    - num_games: int - The number of games to play
    - num_simulations: int - The number of simulations to run for each move

    A high number of simulations leads to better training data, but increases the time it takes to generate the data.

    Returns:
    - list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] - The training data

    The returned list is on the form:
    [(state, probability_target, reward), (state, probability_target, reward), ...]

    """

    alphazero_mcts = AlphaZero()
    nn.to(alphazero_mcts.device)
    training_data = []

    for i in range(num_games):
        new_training_data = play_alphazero_game(alphazero_mcts, nn, num_simulations)
        if (i + 1) % 50 == 0:
            print(f"Game {i + 1} finished")
        training_data.extend(new_training_data)

    return training_data


