"""
This file instantiates the alphazero_training class,
and generates training data by playing games with the alphazero agent.
"""


import torch
import torch.multiprocessing as mp
import time
from tqdm import tqdm

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
    training_data = [
        (
            state,
            probability_visits,
            torch.tensor(
                [rewards[i % 2]], dtype=torch.float, device=alphazero_mcts.device
            ),
        )
        for i, (state, probability_visits) in enumerate(game_data)
    ]

    return training_data


def generate_training_data(nn: NeuralNetwork, num_games: int, num_simulations: int = 100) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    - tuple[torch.Tensor, torch.Tensor, torch.Tensor] - The training data

    Instead of returning a list of tuples, we are just returning three huge tensors.

    """
    alphazero_mcts = AlphaZero()
    nn.to(alphazero_mcts.device)
    training_data = []

    try:
        print(f"Generating training data with {mp.cpu_count()} threads...")
        start_time = time.time()
        with mp.Pool(mp.cpu_count()) as pool:
            result_list = list(tqdm(pool.starmap(play_alphazero_game, [(alphazero_mcts, nn, num_simulations) for _ in range(num_games)])))
        end_time = time.time()
        print(f"Generated training data with {mp.cpu_count()} threads in {end_time - start_time:.2f} seconds.")

        # Process results only if data generation was successful
        for i in range(len(result_list)):
            training_data.extend(result_list[i])

        num_actions = alphazero_mcts.game.num_distinct_actions()
        states = [item[0] for item in training_data]
        probabilities = [item[1] for item in training_data]
        rewards = [item[2] for item in training_data]

        state_tensors = torch.cat(states, dim=0)
        probability_tensors = torch.cat(probabilities, dim=0).reshape(-1, num_actions)
        reward_tensors = torch.cat(rewards, dim=0).reshape(-1, 1)

        return state_tensors, probability_tensors, reward_tensors

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Terminating training data generation...")
        raise
    # pool = None # Initialize pool to None, so we can terminate it if KeyboardInterrupt is raised.
    # alphazero_mcts = AlphaZero()
    # nn.to(alphazero_mcts.device)
    # training_data = []

    # try:
    #     print(f"Generating training data with {mp.cpu_count()} threads...")
    #     start_time = time.time()
    #     pool = mp.Pool(mp.cpu_count())
    #     result_list = list(tqdm(pool.starmap(play_alphazero_game, [(alphazero_mcts, nn, num_simulations) for _ in range(num_games)])))
    #     end_time = time.time()
    #     print(f"Generated training data with {mp.cpu_count()} threads in {end_time - start_time:.2f} seconds.")
        
    # except KeyboardInterrupt:
    #     print("KeyboardInterrupt: Terminating training data generation...")
    #     if pool is not None:
    #         pool.terminate() # Terminates all processes in the pool.
    #         pool.join() # Waits for all processes to finish.
    #         pool.close()
    #     raise # Reraise the KeyboardInterrupt, enables parent process to do its cleanup-process as well.
    
    # for i in range(len(result_list)):
    #         training_data.extend(result_list[i])

    # num_actions = alphazero_mcts.game.num_distinct_actions()
    # states = [item[0] for item in training_data]
    # probabilities = [item[1] for item in training_data]
    # rewards = [item[2] for item in training_data]

    # state_tensors = torch.cat(states, dim=0)
    # probability_tensors = torch.cat(probabilities, dim=0).reshape(-1, num_actions)
    # reward_tensors = torch.cat(rewards, dim=0).reshape(-1, 1)

    # return state_tensors, probability_tensors, reward_tensors

