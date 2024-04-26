"""
Selection, expansion & evaluation, backpropagation.
These methods are standard for AlphaZero.
"""

import torch
from src.alphazero.node import Node
from src.utils.nn_utils import forward_state
from src.utils.tensor_utils import normalize_policy_values, normalize_policy_values_with_noise
from src.utils.game_context import GameContext
from src.utils.random_utils import generate_dirichlet_noise


def vectorized_select(node: Node, c: float) -> Node: # OPTIMIZATION for GPU, great speedup is expected when number of children is large.
    """
    Select stage of MCTS.
    Go through the game tree, layer by layer.
    Chooses the node with the highest UCB-score at each layer.
    Returns a leaf node.
    """
    while node.has_children():
        visits = torch.tensor([child.visits for child in node.children], dtype=torch.float)
        
        values = torch.tensor([child.value for child in node.children], dtype=torch.float)
        
        policy_values = torch.tensor([child.policy_value for child in node.children], dtype=torch.float)
        parent_visits_sqrt = torch.tensor(node.visits, dtype=torch.float).sqrt_()

        # Compute PUCT for all children in a vectorized manner
        Q = torch.where(visits > 0, values / visits, torch.zeros_like(visits))
        U = c * policy_values * parent_visits_sqrt / (1 + visits)
        puct_values = Q + U

        max_puct_index = torch.argmax(puct_values).item() # Find the index of the child with the highest PUCT value
        node = node.children[max_puct_index] # Return the best child node based on PUCT value
    
    return node

def evaluate(node: Node, context: GameContext) -> float:
    """
    Neural network evaluation of the state of the input node.
    Will not be run on a leaf node (terminal state)
    """
    policy, value = forward_state(node.state, context)
    return policy, value

def expand(node: Node, nn_policy_values: torch.Tensor) -> None:
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

def dirichlet_expand(context: GameContext, node: Node, nn_policy_values: torch.Tensor, alpha: float, epsilon: float) -> None:
    """
    TODO: Update this docstring
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
    noise = generate_dirichlet_noise(context, len(legal_actions), alpha)
    normalize_policy_values_with_noise(nn_policy_values, legal_actions, noise, epsilon)

    for action in legal_actions:  # Add the children with correct policy values
        new_state = node.state.clone()
        new_state.apply_action(action)
        node.children.append(
            Node(node, new_state, action, nn_policy_values[action])
        )

def backpropagate(node: Node, result: torch.Tensor) -> None:
    """
    Return the results all the way back up the game tree.
    """
    node.visits += 1
    if node.parent != None:
        node.value += result
        backpropagate(node.parent, -result)