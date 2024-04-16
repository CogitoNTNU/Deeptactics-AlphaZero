import torch
import pyspiel
from torch.distributions.dirichlet import Dirichlet
from icecream import ic
from src.play_vs_alphazero import main

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# main()


import torch
from torch.nn import CrossEntropyLoss

cross_entropy = CrossEntropyLoss()
# Simulated neural network output for 5 classes


nn_output = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)

def training_loop(nn_output):

    # Selecting the probabilities for x3 and x5 as specified
    selected_nn = nn_output[[2, 4]]

    # Applying softmax to convert raw scores to probabilities

    # selected_nn = torch.softmax(selected_nn, dim=0)

    # The target distribution for x3 and x5
    target_distribution = torch.tensor([0.2, 0.8])

    loss = cross_entropy(selected_nn, target_distribution)
    loss.backward()

    # Displaying the initial loss
    ic(loss.item(), nn_output.grad)

    with torch.no_grad():
        nn_output -= 0.1 * nn_output.grad
        nn_output.grad.zero_()


ic(torch.softmax(nn_output[[2, 4]], dim=0))
for _ in range(100):
    training_loop(nn_output)

ic(nn_output)
ic(torch.softmax(nn_output, dim=0))
ic(torch.softmax(nn_output[[2, 4]], dim=0))