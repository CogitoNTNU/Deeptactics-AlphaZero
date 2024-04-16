from src.neuralnet.neural_network import NeuralNetwork
from src.alphazero.alphazero import play_alphazero
from src.alphazero.alphazero_training import train_alphazero
from src.alphazero.alphazero_training import AlphaZero
import torch
import pyspiel

# play_alphazero()

# for i in range(1):
#      train_alphazero(50, 100)
#      print(f'Training session {i} finished!')

# Neural network gives 5 outputs:
## [x1, x2, x3, x4, x5]

"""
Target is [0.2, 0.8] for x3 and x5

Idea:
Pick nn[2, 4] and perforcm cross entropy loss
with target [0.2, 0.8]
Should see that the loss goes down.


"""

alphazero = AlphaZero()
alphazero = alphazero.to(alphazero.device)
nn = NeuralNetwork().to(alphazero.device)
state = alphazero.game.new_initial_state()
for _ in range(10):
     alphazero.run_simulation(state, nn, 1)
