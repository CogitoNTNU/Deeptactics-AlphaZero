from src.alphazero.node import Node
from src.utils.game_context import GameContext
from src.alphazero.tree_search_methods.select import vectorized_select
from src.alphazero.tree_search_methods.evaluate import evaluate
from src.alphazero.tree_search_methods.expand import expand
from src.alphazero.tree_search_methods.backpropagate import backpropagate


class AlphaZero:

    def __init__(self, context: GameContext):
        self.context = context
        self.shape = [1] + context.game.observation_tensor_shape()
        self.c = 4. # Exploration constant

    def run_simulation(self, state, num_simulations=800): # Num-simulations 800 is good for tic-tac-toe
        """
        Selection, expansion & evaluation, backpropagation.

        """
        root_node = Node(parent=None, state=state, action=None, policy_value=None)
        policy, value = evaluate(root_node.state.observation_tensor(), self.shape, self.context.nn, self.context.device)
        print("Root node value: ", value)

        for _ in range(num_simulations):  # Do selection, evaluation & expansion, backpropagation

            node = vectorized_select(root_node, self.c)
            
            if not node.state.is_terminal():
                policy, value = evaluate(node.state.observation_tensor(), self.shape, self.context.nn, self.context.device)
                value = -value
                expand(node, policy)
            
            else:
                player = node.parent.state.current_player()
                value = node.state.returns()[player]
                
            backpropagate(node, value)
        
        for child in root_node.children:
            print(f'Action: {child.action}, Visits: {child.visits}, Value: {child.value}, Value type: {type(child.value)}, Policy Value: {child.policy_value}, Policy type: {type(child.policy_value)}')
        
        chosen_action = max(root_node.children, key=lambda node: node.visits).action
        print(f"Chosen action: {chosen_action}")

        return chosen_action
        
        
def alphazero_self_play(context: GameContext, num_simulations=800):
    """
    For testing purposes.
    The agent plays against itself until it reaches a terminal state.
    """
    
    alphazero = AlphaZero(context=context)
    state = alphazero.context.get_initial_state()
    
    while (not state.is_terminal()):
        action = alphazero.run_simulation(state, num_simulations)
        state.apply_action(action)
        print(state)
    
    print("Game over!")
    print("Player 1 score: ", state.returns()[0], "\nPlayer 2 score: ", state.returns()[1])

