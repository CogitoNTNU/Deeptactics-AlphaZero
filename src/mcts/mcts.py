import numpy as np
import pyspiel
from src.mcts.node import Node

class Mcts:
    
    def __init__(self):
        self.c = 1.41

    def ucb(self, node: Node) -> int:
        if node.visits == 0:
            return np.inf
        return node.value / node.visits + self.c * np.sqrt(
            np.log(node.parent.visits) / node.visits
        )

    def select(self, node: Node) -> Node:
        """
        Select stage of MCTS.
        Go through the game tree, layer by layer.
        Chooses the node with the highest UCB-score at each layer.
        Returns a
        """
        highest_ucb = -np.inf
        best_node: Node = None
        current_node = node

        while current_node.has_children():
            for child in current_node.children:
                current_ucb = self.ucb(child)
                if current_ucb > highest_ucb:
                    highest_ucb = current_ucb
                    best_node = child
            current_node = best_node
            highest_ucb = -np.inf
        return current_node

    def expand(self, node: Node) -> None:
        """
        Optional stage in the MCTS algorithm.
        If you select a leaf node, this method will not be run.

        You expand once per node, you expand by adding all possible children to the children list.
        """
        legal_actions = node.state.legal_actions()
        for action in legal_actions:
            new_state = node.state.clone()
            new_state.apply_action(action)
            node.children.append(Node(node, new_state, action))

    def simulate(self, node: Node):
        """
        Simulate random moves until you reach a leaf node (A conclusion of the game)
        """
        simulation_state = node.state.clone()
        while not (simulation_state.is_terminal()):
            action = np.random.choice(simulation_state.legal_actions())
            simulation_state.apply_action(action)
        return simulation_state.returns()[(node.state.current_player() + 1) & 1]

    def backpropagate(self, node: Node, result: int):
        """
        Return the results all the way back up the game tree.
        """
        node.visits += 1
        if node.parent != None:
            node.value += result
            self.backpropagate(node.parent, -result)

    def run_simulation(self, state, num_simulations=1_000):
        """
        Simulate a game to its conclusion.
        Random moves are selected all the way.
        """
        root_node = Node(None, state, None)
        for _ in range(num_simulations):
            node = self.select(root_node)
            if not node.state.is_terminal() and not node.has_children():
                self.expand(node)  # creates all children
                winner = self.simulate(node)
            else:
                player = node.parent.state.current_player()
                winner = node.state.returns()[player]
            self.backpropagate(node, winner)

        print("num visits\t", [node.visits for node in root_node.children])
        print("actions\t\t", [node.action for node in root_node.children])
            
        return max(root_node.children, key=lambda node: node.visits).action


if __name__ == "__main__":
    game = pyspiel.load_game("tic_tac_toe")
    mcts = Mcts()
    state = game.new_initial_state()

    while not state.is_terminal():
        action = mcts.run_simulation(state, 1_000)
        print("best action\t", action, "\n")
        state.apply_action(action)
        print(state)
        print()
        
    print(state.returns())

