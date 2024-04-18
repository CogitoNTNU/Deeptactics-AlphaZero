import pyspiel
import torch
from torch import multiprocessing as mp

from src.alphazero.alphazero_play_agent import play_alphazero
from src.alphazero.alphazero_train_model import train_alphazero_model
from src.play_vs_alphazero import main as play_vs_alphazero

if __name__ == '__main__': # Needed for multiprocessing to work
     test_overfit = False
     train = True
     self_play = False
     play = False
     mp.set_start_method('spawn')
     if test_overfit:
          train_alphazero_model(
               num_games=1,
               num_simulations=1000,
               epochs=1000,
               batch_size=16,
               model_path=None
          )

     if train:
          for i in range(20):
               train_alphazero_model(
                    num_games=24,
                    num_simulations=5,
                    epochs=2,
                    batch_size=16,
                    model_path=None
               )
               print(f'Training session {i} finished!')

     if self_play:
          play_alphazero("./models/good_nn")

     if play:
          play_vs_alphazero("./models/good_nn")
