from src.neuralnet.neural_network import NeuralNetwork
from src.alphazero.agents.alphazero_play_agent import AlphaZero
from src.utils.game_context import GameContext

def player(state):
    action = None
    while True:
        print("Choose an action: ", state.legal_actions())
        action = input()
        try:
            action = int(action)
            if action in state.legal_actions():
                break
            else:
                print("Invalid move, try again!")
        except ValueError:
            print("Invalid move, try again!")
    state.apply_action(action)


def ai(state, alphazero: AlphaZero):
    action = alphazero.run_simulation(state, num_simulations = 1000)
    state.apply_action(action)


def play_game(player1, player2, state, alphazero: AlphaZero, first: bool):
    
    if not first:
        player2(state, alphazero)
    
    while not state.is_terminal():
        print('~~~~~~~~~~~~~~~ PLAYER 1 ~~~~~~~~~~~~~~~~')
        print(state)
        player1(state)
        print('~~~~~~~~~~~~~~~ PLAYER 2 ~~~~~~~~~~~~~~~~')
        print(state)
        if state.is_terminal():
            break
        player2(state, alphazero)
        if state.is_terminal():
            print('~~~~~~~~~~~~~~~ PLAYER 1 ~~~~~~~~~~~~~~~~')
            print(state)


def main(context: GameContext, first: bool):
    alphazero = AlphaZero(context=context)
    state = alphazero.context.game.new_initial_state()
    play_game(player, ai, state, alphazero, first)
