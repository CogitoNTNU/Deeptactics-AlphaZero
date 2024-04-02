import numpy as np
import pyspiel
import torch

from src.utils.tensor_utils import normalize_policy_values
from src.utils.nn_utils import forward_state
from src.alphazero.node import Node
from src.neuralnet.neural_network import NeuralNetwork


class AlphaZero:
    def __init__(self):
        self.c = 1.41
        self.game = pyspiel.load_game("tic_tac_toe")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def PUCT(self, node: Node) -> float:
        if node.visits == 0:
            Q = 0  # You don't know the value of a state you haven't visited. Get devision error
        else:
            Q = node.value / node.visits  # Take the average

        U = (self.c * node.policy_value * np.sqrt(node.parent.visits) / (1 + node.visits))

        PUCT = Q + U

        return PUCT

    

    # @profile
    def vectorized_select(self, node: Node) -> Node: # OPTIMIZATION for GPU, great speedup is expected when number of children is large.
        """
        Select stage of MCTS.
        Go through the game tree, layer by layer.
        Chooses the node with the highest UCB-score at each layer.
        Returns a
        """

        # Gather data
        while node.has_children():
            visits = torch.tensor([child.visits for child in node.children], device=self.device, dtype=torch.float)
            values = torch.tensor([child.value for child in node.children], device=self.device, dtype=torch.float)
            policy_values = torch.tensor([child.policy_value for child in node.children], device=self.device, dtype=torch.float)
            parent_visits = torch.tensor(node.visits, device=self.device, dtype=torch.float)
            parent_visits_sqrt = torch.sqrt(parent_visits)

            # Compute PUCT for all children in a vectorized manner
            Q = torch.zeros_like(visits)  # Handle division by zero for unvisited nodes
            Q[visits > 0] = values[visits > 0] / visits[visits > 0]
            U = self.c * policy_values * parent_visits_sqrt / (1 + visits)
            puct_values = Q + U

            # Find the index of the child with the highest PUCT value
            max_puct_index = torch.argmax(puct_values).item()
            
            # Return the best child node based on PUCT value
            node = node.children[max_puct_index]
        
        return node

    # @profile
    def evaluate(self, node: Node, neural_network: NeuralNetwork) -> float:
        """
        Neural network evaluation of the state of the input node.
        Will not be run on a leaf node (terminal state)
        """
        shape = self.game.observation_tensor_shape() # The shape of the input tensor (without batch dimension). Returns [3, 3, 3] for tic-tac-toe.
        policy, value = forward_state(node.state, shape, self.device, neural_network)
        return policy, value

    # @profile
    def expand(self, node: Node, nn_policy_values: torch.Tensor) -> None:
        """
        Takes in a node, and adds all children.
        The children need a state, an action and a policy value.
        Will not be run if you encounter an already visited state, and not if the state is terminal.
        
        The policy values is a tensor output from the neural network, and will most likely not be a probability vector.
        Therefore, we normalize the policy values by applying the softmax normalization function to form a probability
        distribution for action selection.
        """
        legal_actions = node.state.legal_actions()
        normalize_policy_values(nn_policy_values, legal_actions)

        for action in legal_actions: # Add the children with correct policy values
            new_state = node.state.clone()
            new_state.apply_action(action)
            node.children.append(Node(node, new_state, action, nn_policy_values[action]))
    
    # @profile
    def backpropagate(self, node: Node, result: float):
        """
        Return the results all the way back up the game tree.
        """
        node.visits += 1
        if node.parent != None:
            node.value += result
            self.backpropagate(node.parent, -result)

    # @profile
    def run_simulation(self, state, neural_network: NeuralNetwork, num_simulations=800):
        """
        Selection, expansion & evaulation, backpropagation.

        """
        root_node = Node(parent=None, state=state, action=None, policy_value=None)  # Initialize root node.

        for _ in range(num_simulations):  # Do the selection, expansion & evaluation, backpropagation

            node = self.vectorized_select(root_node)  # Get desired child node
            if not node.state.is_terminal() and not node.has_children():
                policy, value = self.evaluate(node, neural_network) # Evaluate the node, using the neural network
                # print("Value:", value, "Policy", policy)
                self.expand(node, policy)  # creates all its children
                winner = value
            else:
                player = (node.parent.state.current_player())  # Here state is terminal, so we get the winning player
                if player is None:
                    print("Player is none for some reason...")
                winner = node.state.returns()[player]
            self.backpropagate(node, winner)
        
        return max(root_node.children, key=lambda node: node.visits).action # The best action is the one with the most visits

def play_alphazero():
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    alphazero_mcts = AlphaZero()
    nn = NeuralNetwork().to(alphazero_mcts.device)
    print("Using ", "cuda" if torch.cuda.is_available() else "cpu")
    while (not state.is_terminal()):
        action = alphazero_mcts.run_simulation(state, nn)
        print("best action\t", action, "\n")
        state.apply_action(action)
        print(state)
    nn.save("./models/nn")
    print(state.returns())
    print(type(state))


"""
state.returns() -> rewards for the game.
state.apply_action(action) -> Play an action [number from 0 to 9]
"""
