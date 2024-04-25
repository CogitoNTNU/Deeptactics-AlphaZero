import pyspiel
import torch
from torch import multiprocessing as mp
from src.alphazero.alphazero_training_agent import AlphaZero
from src.neuralnet.neural_network import NeuralNetwork
from src.neuralnet.neural_network_connect_four import NeuralNetworkConnectFour

from src.neuralnet.create_neural_network import create_tic_tac_toe_model, create_connect_four_model
from src.alphazero.alphazero_play_agent import play_alphazero
from src.alphazero.alphazero_train_model import train_alphazero_model
from src.play.play_vs_alphazero import main as play_vs_alphazero


### Idea, make each game generation a longer task.
# Instead of running one function per game generation, run a function that generates multiple games.
# This will make the overhead of creating a new multiprocessing process less significant.


def test_overfit():
     mp.set_start_method('spawn')
     train_alphazero_model(
               alphazero=AlphaZero(game_name="tic_tac_toe"),
               nn=NeuralNetwork(),
               nn_save_path="./models/overfit_nn",
               num_games=1,
               num_simulations=1000,
               epochs=1000,
               batch_size=16,
          )

def train_tic_tac_toe(model_path: str):
     mp.set_start_method('spawn')
     try:
          for i in range(int(1e6)):
               train_alphazero_model(
                    alphazero=AlphaZero(game_name="tic_tac_toe"),
                    nn=NeuralNetwork().load(model_path),
                    nn_save_path=model_path,
                    num_games=360,
                    num_simulations=300,
                    epochs=2,
                    batch_size=32,
               )
               print(f'Training session {i + 1} finished!')
     except KeyboardInterrupt:
          print('Training interrupted!')

def train_connect_four(model_path: str):
     mp.set_start_method('spawn')
     try:
          for i in range(int(1e6)):
               train_alphazero_model(
                    alphazero=AlphaZero(game_name="connect_four"),
                    nn=NeuralNetworkConnectFour().load(model_path),
                    nn_save_path=model_path,
                    num_games=24,
                    num_simulations=500,
                    epochs=2,
                    batch_size=16,
               )
               print(f'Training session {i + 1} finished!')
     except KeyboardInterrupt:
          print('Training interrupted!')


def self_play():
     play_alphazero("./models/great_tic_tac_toe")

def play(game_name: str, model_path: str):
     play_vs_alphazero(game_name,
                       nn = NeuralNetworkConnectFour.load(model_path)
                       )

if __name__ == '__main__': # Needed for multiprocessing to work
     
     # test_overfit()
     # train_tic_tac_toe("./good_exploration_nn")
     train_connect_four("./models/connect_four/initial_test")
     # self_play()
     # play(game_name="connect_four", model_path="./models/connect_four/initial_test")
     # create_tic_tac_toe_model("initial_test")
     # create_connect_four_model("initial_test")
