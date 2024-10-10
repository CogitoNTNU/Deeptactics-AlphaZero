"""
This file instantiates the alphazero_training class,
and generates training data by playing games with the alphazero agent.
"""

import time
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from src.alphazero.agents.alphazero_training_agent import AlphaZero
from src.utils.nn_utils import reshape_pyspiel_state
from src.utils.multi_core_utils import get_play_alphazero_games_arguments

def play_alphazero_game(
    alphazero: AlphaZero, num_simulations: int
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Plays a game using the AlphaZero training agent, and returns a list of training data.
    The number of training data is equal to the number of moves in the game.
    For each move, the training data is as follows:
    (state, probability_target, reward)
    """

    state = alphazero.context.game.new_initial_state()
    game_data = []; move_number = 1

    while not state.is_terminal():
        action, probability_target = alphazero.run_simulation(state, move_number, num_simulations=num_simulations)
        game_data.append((
            reshape_pyspiel_state(state, alphazero.context),
            probability_target
        ))
        state.apply_action(action)
        move_number += 1

    rewards = state.returns()
    training_data = [(
        state,
        probability_target,
        torch.tensor([rewards[i & 1]], dtype=torch.float, device=alphazero.context.device)
        ) for i, (state, probability_target) in enumerate(game_data)
    ]

    return training_data

def play_alphazero_games(
        alphazero: AlphaZero, num_games: int, num_simulations: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Plays a number of games using the AlphaZero training agent, and returns a list of training data.
    The number of training data is equal to the number of moves in each game summed up.
    For each move, the training data is as follows:
    (state, probability_target, reward)
    """
    training_data = []
    for _ in range(num_games):
        training_data.extend(play_alphazero_game(alphazero, num_simulations))
    return training_data
    
def generate_training_data(alphazero: AlphaZero, num_games: int, num_simulations: int = 100) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Takes in a neural network, and generates training data by making the neural network play games against itself.
    The amount of training data is equal to:
    - sum of (number of moves in each game)

    Parameters:
    - alphazero: Alphazero - The AlphaZero agent used to generate the training data
    - num_games: int - The number of games to play
    - num_simulations: int - The number of simulations to run for each move

    A high number of simulations leads to better training data, but increases the time it takes to generate the data.

    Returns:
    - tuple[torch.Tensor, torch.Tensor, torch.Tensor] - The training data

    Instead of returning a list of tuples, we are just returning three huge tensors.
    """
    
    training_data = []

    # start_time = time.time()
    # result_list = [play_alphazero_games(alphazero, num_games, num_simulations)] # Single-threaded
    # end_time = time.time()
    # print(f"Generated training data in {end_time - start_time:.2f} seconds.")

    multicore_args, thread_count = get_play_alphazero_games_arguments(alphazero, num_games, num_simulations)
    try:
        print(f"Generating training data with {thread_count} threads...")
        start_time = time.time()
        with mp.Pool(thread_count) as pool:
            result_list = list(tqdm(pool.starmap(play_alphazero_games, multicore_args)))
        end_time = time.time()
        print(f"Generated training data with {thread_count} threads in {end_time - start_time:.2f} seconds.")

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Terminating training data generation...")
        raise
    
    for i in range(len(result_list)):
        training_data.extend(result_list[i])

    states = [item[0] for item in training_data]
    probabilities = [item[1] for item in training_data]
    rewards = [item[2] for item in training_data]

    state_tensors = torch.cat(states, dim=0)
    probability_tensors = torch.cat(probabilities, dim=0).reshape(-1, alphazero.context.num_actions)
    reward_tensors = torch.cat(rewards, dim=0).reshape(-1, 1)

    return state_tensors, probability_tensors, reward_tensors


