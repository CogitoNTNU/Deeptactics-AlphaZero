from src.mcts.mcts import Mcts
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
# game = pyspiel.load_game("chess")
state = game.new_initial_state()
first_state = state.clone()
mcts = Mcts()
while not state.is_terminal():
    action = mcts.run_simulation(state, 1_000)
    print("best action\t", action, "\n")
    state.apply_action(action)
    print(state)
    # print(np.reshape(np.asarray(state.observation_tensor()), game.observation_tensor_shape()))
    print()
    
