"""
Contains an implementation of the AlphaZero algorithm.
This one is specialized for training.
Instead of using the greedy policy used when playing with an already trained model,
this version has some exploration in it.

Exploration is added by using dirichlet expand on the root node, and also 
by picking a move with temperature probability distribution for the first 30 moves.
Essentially means that instead of taking the move with the highest visit count (greedy),
we take a move with a probability distribution based on the visit counts.

The main method in this file is run_simulation, which simulates a game using the MCTS algorithm.
It returns the best action to take, and the probability distribution of the root node's children visits.
"""

import pyspiel
import torch

from src.alphazero.node import Node
from src.neuralnet.neural_network import NeuralNetwork
from src.utils.nn_utils import forward_state
from src.utils.random_utils import generate_dirichlet_noise, generate_probabilty_target
from src.utils.tensor_utils import (normalize_policy_values,
                                    normalize_policy_values_with_noise)


class AlphaZero(torch.nn.Module):

    def __init__(self, game_name: str = "tic_tac_toe"):
        super(AlphaZero, self).__init__()
        self.game = pyspiel.load_game(game_name)
        self.num_actions = self.game.num_distinct_actions()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.c = torch.tensor(
            4.0, dtype=torch.float, device=self.device
        )  # Exploration constant
        """
        An exploration constant, used when calculating PUCT-values.
        """

        self.a: float = 0.3
        """
        Alpha-value, a parameter in the Dirichlet-distribution.
        """

        self.e: float = 0.75
        """
        Epsilon-value, determines how many percent of ... is determined by PUCT,
        and how much is determined by Dirichlet-distribution.
        """

        self.temperature_moves: int = 30
        """
        Up to a certain number of moves have been played, the move played is taken from a
        probability distribution based on the most visited states.
        After temperature_moves, the move played is deterministically the one visited the most.
        """

    # @profile
    def vectorized_select(
        self, node: Node
    ) -> (
        Node
    ):  # OPTIMIZATION for GPU, great speedup is expected when number of children is large.
        """
        Select stage of MCTS.
        Go through the game tree, layer by layer.
        Chooses the node with the highest UCB-score at each layer.
        Returns a leaf node.
        """

        while (
            node.has_children()
        ):  ## TODO: Check how to force cpu, possibly switch self.device with torch.device("cpu")

            visits = torch.tensor(
                [child.visits for child in node.children],
                device=self.device,
                dtype=torch.float,
            )
            values = torch.tensor(
                [child.value for child in node.children],
                device=self.device,
                dtype=torch.float,
            )
            policy_values = torch.tensor(
                [child.policy_value for child in node.children],
                device=self.device,
                dtype=torch.float,
            )
            parent_visits_sqrt = torch.tensor(
                node.visits, device=self.device, dtype=torch.float
            ).sqrt_()

            # Compute PUCT for all children in a vectorized manner
            Q = torch.where(
                visits > 0, values / visits, 0
            )
            U = self.c * policy_values * parent_visits_sqrt / (1 + visits)
            puct_values = Q + U

            max_puct_index = torch.argmax(
                puct_values
            ).item()  # Find the index of the child with the highest PUCT value
            node = node.children[
                max_puct_index
            ]  # Return the best child node based on PUCT value

        return node

    # @profile
    def evaluate(self, node: Node, neural_network: NeuralNetwork) -> float:
        """
        Neural network evaluation of the state of the input node.
        Will not be run on a leaf node (terminal state)
        """
        shape = (
            self.game.observation_tensor_shape()
        )  # The shape of the input tensor (without batch dimension). Returns [3, 3, 3] for tic-tac-toe.
        policy, value = forward_state(node.state, shape, self.device, neural_network)
        return policy, value

    # @profile
    def expand(self, node: Node, nn_policy_values: torch.Tensor) -> None:
        """
        Takes in a node, and adds all children.
        The children need a state, an action and a policy value.
        Policy values are normalized before being applied to the children.
        """

        legal_actions = node.state.legal_actions()
        normalize_policy_values(nn_policy_values, legal_actions)

        for action in legal_actions:  # Add the children with correct policy values
            new_state = node.state.clone()
            new_state.apply_action(action)
            node.children.append(
                Node(node, new_state, action, nn_policy_values[action])
            )

    # @profile
    def dirichlet_expand(self, node: Node, nn_policy_values: torch.Tensor):
        """
        Method for expanding the root node with dirichlet noise.
        Is only run on the root node.
        The amount of exploration is determined by the epsilon value, high epsilon gives low exploration.

        Parameters:
        - node: Node - The root node of the game tree
        - nn_policy_values: torch.Tensor - The policy values output by the neural network

        NOTE: The nn_policy_values will include the policy values for all actions, not just the legal ones.
        In the following example, we are showing a simplified example where we only have 3 actions, and all
        of them are legal. The policy values are softmaxed, and then dirichlet noise is added to the policy values.

        Example:

        Epsilon = 0.75\n
        NN policy values = [0.3, -0.1, 0.42]\n
        softmaxed NN policy values = [0.3574, 0.2396, 0.4030]\n
        Dirichlet noise = [0.5, 0.2, 0.3]\n
        Final policy values = 0.75 * [0.3574, 0.2396, 0.4030] + 0.25 * [0.5, 0.2, 0.3]

        -> [0.3931, 0.2297, 0.3772]
        """

        legal_actions = node.state.legal_actions()
        noise = generate_dirichlet_noise(len(legal_actions), self.a, self.device)
        normalize_policy_values_with_noise(
            nn_policy_values, legal_actions, noise, self.e
        )

        for action in legal_actions:  # Add the children with correct policy values
            new_state = node.state.clone()
            new_state.apply_action(action)
            node.children.append(
                Node(node, new_state, action, nn_policy_values[action])
            )

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
    def run_simulation(
        self, state: pyspiel.State, neural_network: NeuralNetwork, move_number: int, num_simulations: int = 800
    ) -> tuple[int, torch.Tensor]:
        """
        Selection, expansion & evaluation, backpropagation.
        Returns an action to be played.
        """

        root_node = Node(
            parent=None, state=state, action=None, policy_value=None
        )  # Initialize root node, and do dirichlet expand to get some exploration
        policy, value = self.evaluate(root_node, neural_network)
        self.dirichlet_expand(root_node, policy)

        for _ in range(
            num_simulations - 1
        ):  # Do the selection, expansion & evaluation, backpropagation

            node = self.vectorized_select(root_node)  # Get desired child node
            if not node.state.is_terminal() and not node.has_children():
                policy, value = self.evaluate(
                    node, neural_network
                )  # Evaluate the node, using the neural network
                self.expand(node, policy)  # creates all its children
                winner = value
            else:
                player = (
                    node.parent.state.current_player()
                )  # Here state is terminal, so we get the winning player
                winner = node.state.returns()[player]
            self.backpropagate(node, winner)

        normalized_root_node_children_visits = generate_probabilty_target(root_node, self.num_actions, self.device)

        # if move_number > self.temperature_moves:
        return (
            max(root_node.children, key=lambda node: node.visits).action,
            normalized_root_node_children_visits,
        )  # The best action is the one with the most visits
        # else:
        #     probabilities = torch.softmax(normalized_root_node_children_visits, dim=0) # Temperature-like exploration
        #     return root_node.children[torch.multinomial(probabilities, num_samples=1).item()].action, normalized_root_node_children_visits

