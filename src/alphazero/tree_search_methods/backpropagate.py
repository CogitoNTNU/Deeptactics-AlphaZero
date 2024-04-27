import torch
from src.alphazero.node import Node

one_tensor = torch.tensor(1.)
value_tensor = torch.tensor(0.)

def update_child_node(node: Node, value: float, add: bool):
    """
    Update the value of the child node.
    """
    node.visits += 1
    node.value += value
    parent = node.parent

    index = parent.children.index(node)
    parent.children_visits[index].add_(one_tensor)
    if add:
        parent.children_values[index].add_(value_tensor)
    else:
        parent.children_values[index].sub_(value_tensor)
    return node.parent

def backpropagate(node: Node, value: float) -> None:
    """
    Return the results all the way back up the game tree.
    """
    value_tensor.fill_(value)
    while node.parent != None:
        node = update_child_node(node, value, add=True)
        if node.parent == None:
            break
        node = update_child_node(node, -value, add=False)

    node.visits += 1