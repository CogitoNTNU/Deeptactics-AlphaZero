import copy as copy
import random
import time

import numpy as np
import pyspiel
from open_spiel.python import games


def spiel_chess(moves, state):
    move = 0

    while moves > move:
        action = random.choice(state.legal_actions(state.current_player()))
        # action_string = state.action_to_string(state.current_player(), action)
        # print("Player ", state.current_player(), ", randomly sampled action: ",
        #         action_string)
        state.apply_action(action)
        move += 1
        #print(state.information_state_tensor())
        
        
        if state.is_terminal():
            break
        noob = np.reshape(np.asarray(state.observation_tensor()), [20, 8, 8])


def function_time(func, moves, env):
    start_time = time.perf_counter()
    func(moves, env)
    end_time = time.perf_counter()
    return end_time - start_time


if __name__ == "__main__":
    # normal python chess
    env = gym.make("TicTacToe")
    chess_games = 10
    pytime = 0
    moves = 200

    for i in range(chess_games):
        board = chess.Board()
        board = env
        pytime += function_time(python_chess, moves, board)

    print(f"python chess: {pytime}")
    # Openspiel chess
    game = pyspiel.load_game("chess")
    shape = game.observation_tensor_shape()
    spiel_time = 0

    for i in range(chess_games):
        state = game.new_initial_state()
        spiel_time += function_time(spiel_chess, moves, state)

    print(f"Spiel chess time: {spiel_time}")

    print(f"Spiel is {pytime/spiel_time} times faster")

    