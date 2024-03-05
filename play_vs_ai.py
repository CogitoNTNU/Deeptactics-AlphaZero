from mcts import Mcts
import pyspiel


def player(state):
    action = None
    while True:
        print("Choose a action: ", state.legal_actions())
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


def ai(state):
    action = mcts.run_simulation(state, 1000)
    state.apply_action(action)


def play_game(player1, player2, state):
    while not state.is_terminal():
        print(state)
        player1(state)
        print(state)
        if state.is_terminal():
            break
        player2(state)


if __name__ == "__main__":
    mcts = Mcts()
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    play_game(ai, player, state)
