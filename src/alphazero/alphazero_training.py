import numpy as np
import pyspiel
import torch

from src.alphazero.node import Node
from src.neuralnet.neural_network import NeuralNetwork

### Implement dirichlet noise for root node
### Implement temperature for move selection


class AlphaZeroTraining:
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

    def select(self, node: Node) -> Node:
        """
        Select stage of MCTS.
        Go through the game tree, layer by layer.
        Chooses the node with the highest UCB-score at each layer.
        Returns a
        """

        highest_puct = -torch.inf  # Initialize
        best_node: Node = None
        current_node = node

        while current_node.has_children():
            for child in current_node.children:
                current_puct = self.PUCT(child)
                if current_puct > highest_puct:
                    highest_puct = current_puct
                    best_node = child
            current_node = best_node
            highest_puct = -torch.inf
        return current_node

    def evaluate(self, node: Node, neural_network: NeuralNetwork) -> float:
        """
        Neural network evaluation of the state of the input node.
        Will not be run on a leaf node (terminal state)
        """
        state = node.state
        shape = self.game.observation_tensor_shape()
        state_tensor = torch.reshape(torch.tensor(state.observation_tensor(), device=self.device), shape).unsqueeze(0)
        policy, value = neural_network.forward(state_tensor)
        return value.squeeze(0), policy.squeeze(0)


    def expand(self, node: Node, nn_policy_values: torch.Tensor) -> None:
        """
        Takes in a node, and adds all children.
        The children need a state, an action and a polic value.
        Will not be run if you encounter an already visited state, and not if the state is terminal.
        """
        legal_actions = node.state.legal_actions()
        policy_values = torch.zeros_like(nn_policy_values, device=self.device)
        for legal_action in legal_actions:
            policy_values[legal_action] = nn_policy_values[legal_action]

        short_tensor = torch.zeros(len(legal_actions))
        j = 0
        for i in range(len(policy_values)):
            if policy_values[i] != 0:
                short_tensor[j] = policy_values[i]
                j += 1
        
        
        short_tensor = torch.softmax(short_tensor, dim=0)
        # add dirichlet noise
        short_tensor.add_(torch.distributions.dirichlet.Dirichlet(torch.tensor([0.03] * len(short_tensor))).sample())

        short_tensor_index = 0
        for i in range(len(policy_values)):
            if policy_values[i] != 0:
                policy_values[i] = short_tensor[short_tensor_index]
                short_tensor_index += 1
                
        

        for action in node.state.legal_actions():
            new_state = node.state.clone()
            new_state.apply_action(action)
            node.children.append(Node(node, new_state, action, policy_values[action]))

    def backpropagate(self, node: Node, result: float):
        """
        Return the results all the way back up the game tree.
        """
        node.visits += 1
        if node.parent != None:
            node.value += result
            self.backpropagate(node.parent, -result)

    def run_simulation(self, state, neural_network: NeuralNetwork, num_simulations=800):
        """
        Selection, expansion & evaulation, backpropagation.
        """
        root_node = Node(parent=None, state=state, action=None, policy_value=None)  # Initialize root node.

        for _ in range(num_simulations): # Do the selection, expansion & evaluation, backpropagation

            node = self.select(root_node)  # Get desired childnode
            if not node.state.is_terminal() and not node.has_children():
                value, policy = self.evaluate(node, neural_network)
                # Evaluate the node,
                # print("Value:", value, "Policy", policy)
                self.expand(node, policy)  # creates all its children
                winner = value
            else:
                player = (node.parent.state.current_player())  # Here state is terminal, so we get the winning player
                if player is None:
                    print("Player is none for some reason...")
                winner = node.state.returns()[player]
            self.backpropagate(node, winner)

        # print("num visits\t", [node.visits for node in root_node.children])
        # print("policy values\t", [node.policy_value for node in root_node.children])
        # print("actions\t\t", [node.action for node in root_node.children])
        
        return max(root_node.children, key=lambda node: node.visits).action  
        # The best action is the one with the most visits


def play_alphazero():
    game = pyspiel.load_game("tic_tac_toe")
    # game = pyspiel.load_game("chess")
    state = game.new_initial_state()
    first_state = state.clone()
    alphazero_mcts = AlphaZeroTraining()
    nn = NeuralNetwork().to(alphazero_mcts.device)
    print("Using ", "cuda" if torch.cuda.is_available() else "cpu")
    while (not state.is_terminal()):
        action = alphazero_mcts.run_simulation(state, nn)
        print("best action\t", action, "\n")
        state.apply_action(action)
        print(state)
    nn.save("./models/nn")
    print(state.returns())


"""
state.returns() -> rewards for the game.
state.apply_action(action) -> Play an action [number from 0 to 9]
"""
