import torch
import pyspiel
from torch.distributions.dirichlet import Dirichlet
from icecream import ic
from src.play.play_vs_alphazero import main
import src.alphazero.agents.alphazero_training_agent as az
import src.alphazero.debug.simulate_state as ss


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

ic(az.__doc__)
tex = torch.zeros(3)
ic(tex)

src = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
index = torch.tensor([[0, 1, 0], [1, 1, 1]])
ic(src.dtype, index.dtype)
bob = torch.zeros((2, 3)).scatter_add_(0, index, src)
ic(bob)

def add(a, b = 3):
    return a + b

ic(add(2, 4))

vecz1 = torch.tensor([1, 2, 3])
vecz2 = torch.tensor([4, 5, 6])
vecz3 = torch.tensor([7, 8, 9])
vecz4 = torch.tensor([10, 11, 12])

stacked = torch.cat([vecz1, vecz2, vecz3, vecz4], dim=0)
ic(stacked)


"""
for utfordring in range(3):
    with konkrete_situasjoner and refleksjoner:
        beskriv arbeidsprosess
        beskriv hvordan arbeidsprosess påvirket produkt
"""

from scipy.stats import norm


std_dev = 6
probability = 1 - norm.cdf(std_dev)
once_per = probability ** (-1)
ic(probability)
ic(once_per)
print(torch.cuda.is_available())

gog = [(1, 1, 1) for _ in range(0)]
ic(gog)

gog.extend([])
ic(gog)

gog += [(1, 1, 1) for _ in range(0)]
ic(gog)


game = pyspiel.load_game("connect_four")
state = game.new_initial_state()
shape = game.observation_tensor_shape()
print(state.legal_actions())
ic(game.observation_tensor_shape())
state.apply_action(1)
ic(state)
state_tensor = torch.reshape(torch.tensor(state.observation_tensor(), device=device), shape).unsqueeze(0)
ic(state_tensor)


board_string = """
    .......
    .......
    .......
    ...o...
    x..o...
    xx.ooxx
    """

rows = board_string.strip().split('\n')
for string in reversed(rows):
    print(string)

ss.main()
