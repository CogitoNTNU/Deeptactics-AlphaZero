"""
The main file for the AlphaZero algorithm.
This file takes in a neural network, and trains it using the AlphaZero algorithm.
"""

import torch
from torch import optim
import torch.nn.functional as F

from src.alphazero.alphazero_training_agent import AlphaZero
from src.neuralnet.neural_network import NeuralNetwork
from src.alphazero.alphazero_generate_training_data import generate_training_data


def train_alphazero_model(num_games: int, num_simulations: int, epochs: int, batch_size: int = 1, model_path: str = None):
    """
    Parameters:
    - num_games: The number of games to play, and use for training.
    - num_simulations: The number of simulations to run in the MCTS algorithm.
    - epochs: The number of epochs to train the neural network on the training data.
    - batch_size: The batch size to use when training the neural network.
    - model_path: The path to load the model from. If None, a new model is created.

    Returns:
    - None, but saves a (hopefully) stronger model to the model_path.

    The method generates training data by playing num_games using the AlphaZero agent.
    At each move in all games, a simulation is done num_simulations times.
    The training data is then used to train the neural network.
    The number of times you train over the dataset is determined by the epochs parameter.
    Batch size is the number of training samples to use for gradient calculation.
    """

    alphazero_mcts = AlphaZero()
    
    if model_path:
        nn = NeuralNetwork().load(model_path).to(alphazero_mcts.device)
    else:
        nn = NeuralNetwork().to(alphazero_mcts.device)
        model_path = "./models/test_nn"
    
    optimizer = optim.Adam(
        nn.parameters(), lr=1e-4, weight_decay=1e-4
    )  # Weight decay is L2 regularization
    
    state_tensors, probability_tensors, reward_tensors = generate_training_data(
        nn, num_games, num_simulations
    )
    
    num_samples = state_tensors.size(0)
    
    last_batch_size = num_samples % batch_size
    num_steps = num_samples // batch_size + (1 if last_batch_size > 0 else 0)
    
    try:    

        for epoch in range(epochs):
            policy_loss_tot = 0
            value_loss_tot = 0
            total_loss = 0

            indices = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                
                batch_state_tensor = state_tensors[indices[i : i + batch_size]]
                batch_probability_tensor = probability_tensors[indices[i : i + batch_size]]
                batch_reward_tensor = reward_tensors[indices[i : i + batch_size]]
                
                optimizer.zero_grad()  # Reset gradients
                policy_pred, value_pred = nn.forward(batch_state_tensor)

                # Calculate loss
                value_loss = F.mse_loss(value_pred, batch_reward_tensor)
                policy_loss = F.cross_entropy(policy_pred, batch_probability_tensor)
                loss = policy_loss + value_loss

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                policy_loss_tot += policy_loss.item(); value_loss_tot += value_loss.item(); total_loss += loss.item()
                

            print(
                f"Epoch {epoch+1}, Total Loss: {total_loss / num_steps}, Policy Loss: {policy_loss_tot / num_steps}, Value Loss: {value_loss_tot / num_steps}"
            )

        nn.save(model_path)
        print("\nModel saved!")

    except KeyboardInterrupt:
        nn.save(model_path)
        print("\nModel saved!")
        raise KeyboardInterrupt
