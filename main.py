import pyspiel
import torch

from src.alphazero.alphazero_play_agent import play_alphazero
from src.alphazero.alphazero_train_model import train_alphazero_model

# play_alphazero()

for i in range(1):
     train_alphazero_model(
          num_games=1,
          num_simulations=100,
          epochs=1000,
          batch_size=10,
          model_path=None
     )
     print(f'Training session {i} finished!')

# Neural network gives 5 outputs:
## [x1, x2, x3, x4, x5]

"""
Target is [0.2, 0.8] for x3 and x5

Idea:
Pick nn[2, 4] and perforcm cross entropy loss
with target [0.2, 0.8]
Should see that the loss goes down.


"""

# alphazero = AlphaZero()
# alphazero = alphazero.to(alphazero.device)
# nn = NeuralNetwork().to(alphazero.device)
# state = alphazero.game.new_initial_state()
# for _ in range(10):
#      alphazero.run_simulation(state, nn, 1)
