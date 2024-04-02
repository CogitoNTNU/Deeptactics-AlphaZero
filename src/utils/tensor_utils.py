"""
File containing helpful functions for manipulating tensors.
The primary motivation behind this file is separation of concerns.

Instead of having small tensor manipulation functions scattered throughout the codebase, we can collect them here,
and possible even reuse them if they are used several places in the codebase.

It is also easier to test these functions in isolation, as they are not dependent on the rest of the codebase.
Additionally, if we want to change the implementation of these functions, we only have to change them in one place.
Optimizations are therefore much easier to implement.
"""

import torch

def normalize_policy_values(nn_policy_values: torch.Tensor, legal_actions: torch.Tensor) -> None:
        """
        Takes in a tensor of policy values output by the neural network. It performs a softmax normalization on the policy values,
        based on the legal actions for the current state. The indices which correspond to illegal actions are kept unchanged.

        Parameters:
        - nn_policy_values: torch.Tensor - The policy values output by the neural network
        - legal_actions: torch.Tensor - Tensor containing the legal actions for the current state

        Returns:
        - None, but modifies the nn_policy_values tensor in place

        Example:\n
        nn_policy_values = torch.tensor([-0.4, 0, 0.7, 0])\n
        legal_actions = torch.tensor([1, 3])\n
        normalize_policy_values(nn_policy_values, legal_actions)\n
        print(nn_policy_values) ## Output: tensor([-0.4, 0.5, 0.7, 0.5])
        """

        legal_policy_values = nn_policy_values[legal_actions] ## Get a tensor which only contains the policy values for the legal actions

        probabilities = torch.softmax(legal_policy_values, dim=0) ## Normalize the policy values to form a probability distribution

        nn_policy_values[legal_actions] = probabilities ## Insert the normalized policy values back into the original tensor
