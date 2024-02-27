import pyspiel
import numpy as np
from open_spiel.python import games
from node import Node

class mcts:

    def __init__(self):
        self.c = 1.41
    
    def ucb(self, node: Node) -> int:
        if node.visits == 0:
            return np.inf
        return node.value/node.visits + self.c * np.sqrt(np.log(node.parent.visits) / node.visits)
    
    def select(self, node):
        highest_ucb = 0
        best_node = None
        for child in node.children:
            current_ucb = self.ucb(child)
            if current_ucb > highest_ucb:
                highest_ucb = current_ucb
                best_node = child
        
        
        
        return best_node

    def expand(self, node, possible_actions):
        pass

    def simulate(self, node):
        pass

    def backpropagate(self, node, result):
        pass
    
    def run_simulation(self, num_simulation):
        pass
        
        
if __name__ == "__main__":
    pass