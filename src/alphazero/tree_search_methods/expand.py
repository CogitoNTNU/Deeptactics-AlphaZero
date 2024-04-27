import torch
from src.alphazero.node import Node
from src.utils.game_context import GameContext
from src.utils.tensor_utils import normalize_policy_values_with_noise
from src.utils.random_utils import generate_dirichlet_noise


# @profile
def expand(node: Node, nn_policy_values: torch.Tensor) -> None:
    """
    Takes in a node, and adds all children.
    The children need a state, an action and a policy value.
    Will not be run if you encounter an already visited state, and not if the state is terminal.
    
    The policy values is a tensor output from the neural network, and will most likely not be a probability vector.
    Therefore, we normalize the policy values by applying the softmax normalization function to form a probability
    distribution for action selection.
    """
    state = node.state
    legal_actions = state.legal_actions()
    nn_policy_values = nn_policy_values.cpu()
    policy_values = torch.softmax(nn_policy_values[legal_actions], dim=0)
    node.set_children_policy_values(policy_values)

    children = node.children
    for action, policy_value in zip(legal_actions, policy_values):
        new_state = state.clone()
        new_state.apply_action(action)
        children.append(Node(node, new_state, action, policy_value))
  

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
    policy_values = nn_policy_values.to("cpu")
    node.set_children_policy_values(policy_values[legal_actions])

    for action in legal_actions:  # Add the children with correct policy values
        new_state = node.state.clone()
        new_state.apply_action(action)
        node.children.append(
            Node(node, new_state, action, policy_values[action])
        )