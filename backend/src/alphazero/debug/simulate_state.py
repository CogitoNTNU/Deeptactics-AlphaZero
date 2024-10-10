import pyspiel
import torch
from src.alphazero.agents.alphazero_play_agent import AlphaZero
from src.neuralnet.neural_network_connect_four import NeuralNetworkConnectFour
from src.utils.game_context import GameContext
from icecream import ic

def replace_char(s: str, ch: str, i: int) -> str: 
    return s[:i] + ch + s[i + 1:]

def string_to_state(game, board_string: str):
    """
    This function is not perfect and will not work for many board states.
    """
    state = game.new_initial_state()
    rows = board_string.strip().split('\n')
    flatten = ''.join(rows)

    if not (flatten.count('x') == flatten.count('o') or flatten.count('x') == flatten.count('o') + 1):
        raise ValueError("Invalid board string")

    while 'x' in flatten or 'o' in flatten:
        
        while 'x' in flatten:

            piece_played = False

            for i in reversed(range(1, len(rows))):
                
                index = rows[i].find('x')
                if index != -1:
                    if rows[i - 1][index] == 'o':
                        state.apply_action(index)
                        rows[i] = replace_char(rows[i], '.', index)
                        piece_played = True
                        break
            
            if piece_played:
                break

            for i in reversed(range(len(rows))):
                index = rows[i].find('x')
                if index != -1:
                    state.apply_action(index)
                    rows[i] = replace_char(rows[i], '.', index)
                    break
            break

        while 'o' in flatten:

            piece_played = False

            for i in reversed(range(1, len(rows))):
                
                index = rows[i].find('o')
                if index != -1:
                    if rows[i - 1][index] == 'x':
                        state.apply_action(index)
                        rows[i] = replace_char(rows[i], '.', index)
                        piece_played = True
                        break
            
            if piece_played:
                break

            for i in reversed(range(len(rows))):
                index = rows[i].find('o')
                if index != -1:
                    state.apply_action(index)
                    rows[i] = replace_char(rows[i], '.', index)
                    break
            break
            
        flatten = ''.join(rows)
    return state

def main():

    game_str = "connect_four"
    save_path = "models/connect_four/initial_test"
    nn = NeuralNetworkConnectFour().load(save_path)
    context = GameContext(game_str, nn, save_path)
    
    alphazero = AlphaZero(context)

    initial_string ="""
.......
.......
.......
.......
.......
.......
    """
    board_string = """
.......
.......
...x...
o..oox.
x..oxxo
xxoooxx
    """
    state = string_to_state(context.game, initial_string)
    ic(state)
    result = alphazero.run_simulation(state)
    print("Simulation result:", result)

