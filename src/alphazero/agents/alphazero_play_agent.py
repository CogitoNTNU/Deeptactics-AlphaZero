import pyspiel
import torch

from src.alphazero.node import Node
from src.utils.game_context import GameContext
from src.alphazero.alphazero_tree_search_methods import vectorized_select, evaluate, expand, backpropagate

class AlphaZero:

    def __init__(self, context: GameContext):
        self.context = context
        self.c = 4.0 # Exploration constant

    def run_simulation(self, state, num_simulations=800): # Num-simulations 800 is good for tic-tac-toe
        """
        Selection, expansion & evaluation, backpropagation.

        """
        root_node = Node(parent=None, state=state, action=None, policy_value=None)  # Initialize root node.
        policy, value = evaluate(root_node, self.context)  # Evaluate the root node
        print("Root node value: ", value)

        for _ in range(num_simulations):  # Do selection, expansion & evaluation, backpropagation

            node = vectorized_select(root_node, self.c)
            
            if not node.state.is_terminal() and not node.has_children():
                policy, value = evaluate(node, self.context) # Evaluate the node, using the neural network
                value = -value
                expand(node, policy)
            else:
                player = node.parent.state.current_player()  # Here state is terminal, so we get the winning player
                value = node.state.returns()[player]
                
            backpropagate(node, value)
        
        for child in root_node.children:
            #print(f'Action: {child.action}, Visits: {child.visits}, Value: {child.value}, Value type: {type(child.value)}, Policy Value: {child.policy_value}, Policy type: {type(child.policy_value)}')
            print(f'Action: {child.action}, Visits: {child.visits}, Value: {round(float(child.value), 2)} Policy Value: {round(child.policy_value, 2)}')
        
        return max(root_node.children, key=lambda node: node.visits).action # The best action is the one with the most visits
        
        
def alphazero_self_play(context: GameContext, num_simulations=800):
    """
    For testing purposes.
    The agent plays against itself until it reaches a terminal state.
    """
    
    alphazero = AlphaZero(context=context)
    state = alphazero.context.get_initial_state()
    
    while (not state.is_terminal()):
        action = alphazero.run_simulation(state, num_simulations)
        print("best action\t", action, "\n")
        state.apply_action(action)
        print(state)
    
    print("Game over!")
    print("Player 1 score: ", state.returns()[0], "\nPlayer 2 score: ", state.returns()[1])



"""
state.returns() -> rewards for the game.
state.apply_action(action) -> Play an action [number from 0 to 8 for tic-tac-toe]
"""