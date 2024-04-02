import pyspiel
# M0nt3 Carl0 Tr33 S3arch
class Node:

    def __init__(self, parent: "Node", state: pyspiel.State, action: int, policy_value: float):

        self.children: list["Node"] = []
        """
        A list of game states you can reach from the current node.
        """
        
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

    def has_children(self) -> bool:
        """
        Returns True if the state has children.
        """
        return len(self.children) > 0

    def __repr__(self) -> str:
        return str(self.state)
    

