from src.alphazero.agents.alphazero_play_agent import AlphaZero
from src.mcts.mcts import Mcts
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

def ai(state, alphazero: AlphaZero | Mcts):
    action = alphazero.run_simulation(state, num_simulations = 1000)
    state.apply_action(action)

def play_mcts_vs_alphazero(context: GameContext, first: bool):
    """
    Lets the MCTS agent play against the AlphaZero agent.
    If first is True, the MCTS agent will play first.
    Both agents are given 1000 simulations before making a move.
    """

    mcts = Mcts()
    alphazero = AlphaZero(context=context)
    state = alphazero.context.game.new_initial_state()

    if not first:
        print('~~~~~~~~~~~~~~~ ALPHAZERO ~~~~~~~~~~~~~~~~')
        print(state)
        ai(state, alphazero)
    
    while not state.is_terminal():
        print('~~~~~~~~~~~~~~~ MONTE CARLO ~~~~~~~~~~~~~~~~')
        print(state)
        ai(state, mcts)
        print('~~~~~~~~~~~~~~~ ALPHAZERO ~~~~~~~~~~~~~~~~')
        print(state)
        if state.is_terminal():
            break
        ai(state, alphazero)
        if state.is_terminal():
            print('~~~~~~~~~~~~~~~ MONTE CARLO ~~~~~~~~~~~~~~~~')
            print(state)

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


def main(context: GameContext, first: bool, mcts: bool = False):

    if mcts:
        play_mcts_vs_alphazero(context, first)
        return
    alphazero = AlphaZero(context=context)
    state = alphazero.context.game.new_initial_state()
    play_game(player, ai, state, alphazero, first)
