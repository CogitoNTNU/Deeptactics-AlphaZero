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

import os
import pyspiel
import torch

from src.alphazero.node import Node
from src.utils.game_context import GameContext

from src.utils.random_utils import generate_probabilty_target
from src.alphazero.tree_search_methods.select import vectorized_select
from src.alphazero.tree_search_methods.evaluate import evaluate
from src.alphazero.tree_search_methods.expand import expand, dirichlet_expand
from src.alphazero.tree_search_methods.backpropagate import backpropagate

class AlphaZero(torch.nn.Module):

    def __init__(self, context: GameContext, c: float = 4.0, alpha: float = 0.3, epsilon: float = 0.25, temperature_moves: int = 30):
        super(AlphaZero, self).__init__()
        
        self.context = context
        """
        Contains useful information like the game, neural network and device.
        """

        self.shape: list[int] = [1] + context.game.observation_tensor_shape()
        """
        The shape which the state tensor must have in order to be compatible with the neural network.
        """

        self.c = c
        """
        An exploration constant, used when calculating PUCT-values.
        """

        self.a = alpha
        """
        Alpha-value, a parameter in the Dirichlet-distribution.
        """

        self.e = epsilon
        """
        Epsilon-value, determines how many percent of ... is determined by PUCT,
        and how much is determined by Dirichlet-distribution.
        """

        self.temperature_moves = temperature_moves
        """
        Up to a certain number of moves have been played, the move played is taken from a
        probability distribution based on the most visited states.
        After temperature_moves, the move played is deterministically the one visited the most.
        """

    def run_simulation(
        self, state: pyspiel.State, move_number: int, num_simulations: int = 800
    ) -> tuple[int, torch.Tensor]:
        """
        Selection, expansion & evaluation, backpropagation.
        Returns an action to be played.
        """
        try: 
            root_node = Node(parent=None, state=state, action=None, policy_value=None)
            policy, value = evaluate(root_node.state.observation_tensor(), self.shape, self.context.nn, self.context.device)
            dirichlet_expand(root_node, policy, self.a, self.e)
            backpropagate(root_node, value)

            for _ in range(num_simulations - 1):  # Selection, evaluation & expansion, backpropagation

                node = vectorized_select(root_node, self.c)

                if not node.state.is_terminal():
                    policy, value = evaluate(node.state.observation_tensor(), self.shape, self.context.nn, self.context.device)
                    expand(node, policy)
                
                else:
                    player = node.parent.state.current_player()
                    value = node.state.returns()[player]
                    
                backpropagate(node, value)

            normalized_root_node_children_visits = generate_probabilty_target(root_node, self.context)

            if move_number > self.temperature_moves:
                return (
                    max(root_node.children, key=lambda node: node.visits).action,
                    normalized_root_node_children_visits
                )
            else:
                masked_values = torch.where(normalized_root_node_children_visits > 0, normalized_root_node_children_visits, torch.tensor(float('-inf'), device=self.context.device))
                probabilities = torch.softmax(masked_values, dim=0) # Temperature-like exploration
                action, probability_target = torch.multinomial(probabilities, num_samples=1).item(), normalized_root_node_children_visits # TODO: Check this line, it really seems like something is wrong with our exploration.
                return action, probability_target
        
        except KeyboardInterrupt:
            print(f'Simulation (run_simulation) interrupted! PID: {os.getpid()}')
            raise
