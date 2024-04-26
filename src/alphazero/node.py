import pyspiel
import torch

# M0nt3 Carl0 Tr33 S3arch
class Node:

    def __init__(self, parent: "Node", state: pyspiel.State, action: int, policy_value: float):

        self.children: list["Node"] = []
        """
        A list of game states you can reach from the current node.
        """
        # TODO: ASSIGN TORCH TENSOR WITH POLICY VALUES TO CHILDREN
        self.parent: 'Node' = parent
        """
        The node representing the state which came before the current node.
        """

        self.state: pyspiel.State = state
        """"
        The current game state.
        """

        self.action = action
        """
        The action which was taken.\n
        This will be a number between 0 and 8.
        """

        self.visits: int = 0
        """
        The number of times this node has been visited.
        """

        self.value: float = 0
        """
        The cumulative reward gained from this game state.
        """
        
        self.policy_value = policy_value
        """
        The value of the policy network from the parent node.
        """

        self.children_policy_values: torch.Tensor = None
        """
        The policy values of the children of the current node.
        """

        self.children_visits: torch.Tensor = None
        """
        The number of visits to the children of the current node.
        """

        self.children_values: torch.Tensor = None
        """
        The values of the children of the current node.
        """


    def has_children(self) -> bool:
        """
        Returns True if the state has children.
        """
        return len(self.children) > 0
    
    def set_children_policy_values(self, policy_values) -> None:
        """
        Sets the policy values of the children of the current node.
        Additionally, initializes the children_visits and children_values tensors.
        """
        self.children_policy_values = policy_values
        self.children_visits = torch.zeros_like(policy_values)
        self.children_values = torch.zeros_like(policy_values)

    def __repr__(self) -> str:
        return str(self.state)
    

