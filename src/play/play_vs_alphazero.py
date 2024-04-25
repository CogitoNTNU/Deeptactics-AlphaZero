from src.neuralnet.neural_network import NeuralNetwork
from src.alphazero.alphazero_play_agent import AlphaZero

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


def ai(state, alphazero: AlphaZero, nn: NeuralNetwork):
    action = alphazero.run_simulation(state, nn, num_simulations = 1000)
    state.apply_action(action)


def play_game(player1, player2, state, alphazero: AlphaZero, nn: NeuralNetwork, first: bool):
    
    if not first:
        player2(state, alphazero, nn)
    
    while not state.is_terminal():
        print('~~~~~~~~~~~~~~~ PLAYER 1 ~~~~~~~~~~~~~~~~')
        print(state)
        player1(state)
        print('~~~~~~~~~~~~~~~ PLAYER 2 ~~~~~~~~~~~~~~~~')
        print(state)
        if state.is_terminal():
            break
        player2(state, alphazero, nn)
        if state.is_terminal():
            print('~~~~~~~~~~~~~~~ PLAYER 1 ~~~~~~~~~~~~~~~~')
            print(state)


def main(game_name: str, nn: NeuralNetwork, first: bool):
    alphazero = AlphaZero(game_name=game_name)
    nn.to(alphazero.device)
    state = alphazero.game.new_initial_state()
    play_game(player, ai, state, alphazero, nn, first)