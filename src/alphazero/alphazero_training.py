import pyspiel
import torch
import torch.optim as optim
from torch.nn import MSELoss, CrossEntropyLoss

from src.alphazero.node import Node
from src.neuralnet.neural_network import NeuralNetwork
from src.utils.nn_utils import forward_state
from src.utils.random_utils import generate_dirichlet_noise
from src.utils.tensor_utils import normalize_policy_values, normalize_policy_values_with_noise



class AlphaZero:

    def __init__(self):
        self.game = pyspiel.load_game("tic_tac_toe")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.c = torch.tensor(1.41, dtype=torch.float, device=self.device) # Exploration constant
        """
        An exploration constant, used when calculating PUCT-values.
        """

        self.a: float = 0.3
        """
        Alpha-value, a parameter in the Dirichlet-distribution.
        """

        self.e: float = 0.75
        """
        Epsilon-value, determines how many percent of ... is determined by PUCT,
        and how much is determined by Dirichlet-distribution.
        """

        self.temperature_moves: int = 30
        """
        Up to a certain number of moves have been played, the move played is taken from a
        probability distribution based on the most visited states.
        After temperature_moves, the move played is deterministically the one visited the most.
        """

    # @profile
    def vectorized_select(self, node: Node) -> Node: # OPTIMIZATION for GPU, great speedup is expected when number of children is large.
        """
        Select stage of MCTS.
        Go through the game tree, layer by layer.
        Chooses the node with the highest UCB-score at each layer.
        Returns a leaf node.
        """

        while node.has_children():
            
            visits = torch.tensor([child.visits for child in node.children], device=self.device, dtype=torch.float)
            values = torch.tensor([child.value for child in node.children], device=self.device, dtype=torch.float)
            policy_values = torch.tensor([child.policy_value for child in node.children], device=self.device, dtype=torch.float)
            parent_visits_sqrt = torch.tensor(node.visits, device=self.device, dtype=torch.float).sqrt_()

            # Compute PUCT for all children in a vectorized manner
            Q = torch.where(visits > 0, values / visits, torch.zeros_like(visits))
            U = self.c * policy_values * parent_visits_sqrt / (1 + visits)
            puct_values = Q + U

            max_puct_index = torch.argmax(puct_values).item() # Find the index of the child with the highest PUCT value
            node = node.children[max_puct_index] # Return the best child node based on PUCT value
        
        return node

    # @profile
    def evaluate(self, node: Node, neural_network: NeuralNetwork) -> float:
        """
        Neural network evaluation of the state of the input node.
        Will not be run on a leaf node (terminal state)
        """
        shape = self.game.observation_tensor_shape() # The shape of the input tensor (without batch dimension). Returns [3, 3, 3] for tic-tac-toe.
        policy, value = forward_state(node.state, shape, self.device, neural_network)
        return policy, value

    # @profile
    def expand(self, node: Node, nn_policy_values: torch.Tensor) -> None:
        """
        Takes in a node, and adds all children.
        The children need a state, an action and a policy value.
        Policy values are normalized before being applied to the children.
        """

        legal_actions = node.state.legal_actions()
        normalize_policy_values(nn_policy_values, legal_actions)

        for action in legal_actions: # Add the children with correct policy values
            new_state = node.state.clone()
            new_state.apply_action(action)
            node.children.append(Node(node, new_state, action, nn_policy_values[action]))
    
    # @profile
    def dirichlet_expand(self, node: Node, nn_policy_values: torch.Tensor):
        """
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
        noise = generate_dirichlet_noise(len(legal_actions), self.a, self.device)
        normalize_policy_values_with_noise(nn_policy_values, legal_actions, noise, self.e)
        
        for action in legal_actions: # Add the children with correct policy values
            new_state = node.state.clone()
            new_state.apply_action(action)
            node.children.append(Node(node, new_state, action, nn_policy_values[action]))

    # @profile
    def backpropagate(self, node: Node, result: float):
        """
        Return the results all the way back up the game tree.
        """
        node.visits += 1
        if node.parent != None:
            node.value += result
            self.backpropagate(node.parent, -result)

    # @profile
    def run_simulation(self, state, neural_network: NeuralNetwork, move_number, num_simulations=800):
        """
        Selection, expansion & evaluation, backpropagation.
        Returns the best action to take.
        """
        root_node = Node(parent=None, state=state, action=None, policy_value=None)  # Initialize root node.
        
        ## Add dirichlet noise to root node policy_values. 
        policy, value = self.evaluate(root_node, neural_network)
        self.dirichlet_expand(root_node, policy)

        for _ in range(num_simulations - 1):  # Do the selection, expansion & evaluation, backpropagation

            node = self.vectorized_select(root_node)  # Get desired child node
            if not node.state.is_terminal() and not node.has_children():
                policy, value = self.evaluate(node, neural_network) # Evaluate the node, using the neural network
                # print("Value:", value, "Policy", policy)
                self.expand(node, policy)  # creates all its children
                winner = value
            else:
                player = (node.parent.state.current_player())  # Here state is terminal, so we get the winning player
                if player is None:
                    print("Player is none for some reason...")
                winner = node.state.returns()[player]
            self.backpropagate(node, winner)

        normalized_root_node_children_visits = [node.visits / (root_node.visits) for node in root_node.children]

        if move_number > self.temperature_moves:
            return max(root_node.children, key=lambda node: node.visits).action, normalized_root_node_children_visits # The best action is the one with the most visits
        else:
            probabilities = torch.softmax(torch.tensor([node.visits for node in root_node.children], dtype=float), dim=0)
            return root_node.children[torch.multinomial(probabilities, num_samples=1).item()].action, normalized_root_node_children_visits
        
        

        
def play_alphazero_game(alphazero_mcts: AlphaZero, nn: NeuralNetwork) -> list[tuple]:

    state = alphazero_mcts.game.new_initial_state()
    shape = alphazero_mcts.game.observation_tensor_shape()
    states, mcts_probability_visits = [], []
    
    
    move_number = 1
    while (not state.is_terminal()):
        action, probability_visits = alphazero_mcts.run_simulation(state, nn, move_number)
        print(len(probability_visits))
        state_tensor = torch.reshape(torch.tensor(state.observation_tensor(), device=alphazero_mcts.device), shape).unsqueeze(0)
        states.append(state_tensor)
        mcts_probability_visits.append(probability_visits)
        print("best action\t", action, "\n")
        state.apply_action(action)
        move_number += 1
    
    winner = state.returns()
    tuple_list = []
    for i in range(len(states)):
        tuple_list.append((states[i], mcts_probability_visits[i], winner[i % 2]))

    return tuple_list # [ (1, 2, 3), (1, 2, 3), (1, 2, 3), ]



    # MSE Loss value: (y-y_hat)^2 - y = actual winner value, y_hat = predicted target value
    


def train_alphazero(num_games: int, epochs: int):

    alphazero_mcts = AlphaZero()
    nn = NeuralNetwork().to(alphazero_mcts.device)
    optimizer = optim.Adam(nn.parameters(), lr=0.001)  # Adjust learning rate as needed

    mse_loss_fn = MSELoss()
    cross_entropy_loss_fn = CrossEntropyLoss()

    training_data = []

    # Generate training data
    for _ in range(num_games):
        new_training_data = play_alphazero_game(alphazero_mcts, nn)
        training_data.extend(new_training_data)

    for epoch in range(epochs):
        
        total_loss = 0
        
        for state_tensor, policy, value in training_data:
            
            optimizer.zero_grad() # Reset gradients
            policy_pred, value_pred = nn.forward(state_tensor) # Forward pass
            print(policy_pred)
            print(value_pred)
            policy_pred.squeeze(0)
            value_pred.squeeze(0)

            # (2, 3, 5)
            # (, 2, 3, 5)
            # (2, 3, 1, 5)

            # Calculate loss
            value_loss = mse_loss_fn(value_pred, torch.tensor([value], dtype=torch.float, device=alphazero_mcts.device))
            # policy_loss = cross_entropy_loss_fn(policy_pred, torch.tensor(policy, dtype=torch.float, device=alphazero_mcts.device).unsqueeze(0))
            # regulization = sum(w^2)
            loss = value_loss # + policy_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item() # Keep track of loss

        print(f'Epoch {epoch+1}, Total Loss: {total_loss}')
    
    
    nn.save("./models/nn")


"""
state.returns() -> rewards for the game.
state.apply_action(action) -> Play an action [number from 0 to 9]
"""