import pyspiel
import torch
from torch import multiprocessing as mp

from src.alphazero.alphazero_play_agent import play_alphazero
from src.alphazero.alphazero_train_model import train_alphazero_model
from src.play_vs_alphazero import main as play_vs_alphazero

### Idea, make each game generation a longer task.
# Instead of running one function per game generation, run a function that generates multiple games.
# This will make the overhead of creating a new multiprocessing process less significant.


def test_overfit():
     train_alphazero_model(
               num_games=10,
               num_simulations=1000,
               epochs=1000,
               batch_size=16,
               model_path=None
          )

def train():
     try:
          for i in range(int(1e6)):
               train_alphazero_model(
                    num_games=24,
                    num_simulations=300,
                    epochs=2,
                    batch_size=16,
                    model_path="./models/good_nn"
               )
               print(f'Training session {i + 1} finished!')
               print(torch.cuda.memory_summary())
               torch.cuda.empty_cache()
     except KeyboardInterrupt:
          print('Training interrupted!')

def self_play():
     play_alphazero("./models/good_nn")

def play():
     play_vs_alphazero("./models/good_nn")

if __name__ == '__main__': # Needed for multiprocessing to work
     mp.set_start_method('spawn')
     # test_overfit()
     train()
     # self_play()
     # play()
