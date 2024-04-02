import torch
from torch.testing import assert_close
from src.utils.tensor_utils import normalize_policy_values


"""
Comment on the use of torch.testing.assert_close:
The torch.testing.assert_close function is used to compare two tensors for approximate equality.
Floating point operations can sometimes lead to small differences in the values of two tensors that should be equal.
Additionally, the function is checking that both tensors have the same shape, dtype, and device.
"""

def test_normalize_policy_values_basic():
    """
    Tests basic functionality of normalize_policy_values to ensure it normalizes
    policy values correctly for specified legal actions.
    """
    nn_policy_values = torch.tensor([-0.4, 0, 0.7, 0], dtype=torch.float)
    legal_actions = torch.tensor([1, 3], dtype=torch.long)
    expected_output = torch.tensor([-0.4, 0.5, 0.7, 0.5])  # Expected softmax values for indices 1 and 3

    normalize_policy_values(nn_policy_values, legal_actions)

    assert_close(nn_policy_values, expected_output)

def test_normalize_policy_values_all_legal():
    """
    Tests the normalize_policy_values function when all actions are legal,
    ensuring the entire tensor is normalized.
    """
    nn_policy_values = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float)
    legal_actions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    expected_output = torch.softmax(nn_policy_values, dim=0)

    normalize_policy_values(nn_policy_values, legal_actions)

    assert_close(nn_policy_values, expected_output)

def test_normalize_policy_values_no_legal():
    """
    Tests the normalize_policy_values function with an empty legal_actions tensor,
    expecting the original nn_policy_values tensor to remain unchanged.
    """
    nn_policy_values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float)
    legal_actions = torch.tensor([], dtype=torch.long)
    expected_output = nn_policy_values.clone()

    normalize_policy_values(nn_policy_values, legal_actions)

    torch.testing.assert_close(nn_policy_values, expected_output)
