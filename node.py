# M0nt3 Carl0 Tr33 S3arch
class Node:

    def __init__(self, parent: "Node", state, action: int):

        self.children: list["Node"] = []
        """
        A list of game states you can reach from the current node.
        """

        self.parent = parent
        """
        The node representing the state which came before the current node.
        """

        self.state = state
        """"
        The current game state.
        """

        self.visits = 0
        """
        The number of times this node has been visited.
        """

        self.value = 0
        """
        The cumulative reward gained from this game state.
        """

        self.action = action
        """
        The action which was taken.\n
        This will be a number between 1 and 9.
        """
