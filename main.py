import pyspiel
import torch
from torch import multiprocessing as mp
from src.alphazero.agents.alphazero_training_agent import AlphaZero
from src.neuralnet.neural_network import NeuralNetwork
from src.neuralnet.neural_network_connect_four import NeuralNetworkConnectFour

from src.neuralnet.create_neural_network import create_tic_tac_toe_model, create_connect_four_model
from src.alphazero.agents.alphazero_play_agent import alphazero_self_play
from src.alphazero.alphazero_train_model import train_alphazero_model
from src.play.play_vs_alphazero import main as play_vs_alphazero
from src.utils.game_context import GameContext


### Idea, make each game generation a longer task.
# Instead of running one function per game generation, run a function that generates multiple games.
# This will make the overhead of creating a new multiprocessing process less significant.


def test_overfit():
     mp.set_start_method('spawn')
     
     overfit_context = GameContext(
          game_name="tic_tac_toe", 
          nn=NeuralNetwork(), 
          save_path="./models/overfit_nn"
     )
     
     train_alphazero_model(
          context=overfit_context,
          num_games=1,
          num_simulations=1000,
          epochs=1000,
          batch_size=16
     )

def train_tic_tac_toe(context: GameContext):
     mp.set_start_method('spawn')

     try:
          for i in range(int(1e6)):
               train_alphazero_model(
                    context=context,
                    num_games=96,
                    num_simulations=100,
                    epochs=3,
                    batch_size=32
               )
               print(f'Training session {i + 1} finished!')
     except KeyboardInterrupt:
          print('Training interrupted!')

def train_connect_four(context: GameContext):
     mp.set_start_method('spawn')
     try:
          for i in range(int(1e6)):
               train_alphazero_model(
                    context=context,
                    num_games=48,
                    num_simulations=100,
                    epochs=3,
                    batch_size=256,
               )
               print(f'Training session {i + 1} finished!')
     except KeyboardInterrupt:
          print('Training interrupted!')


def self_play(context: GameContext):
     alphazero_self_play(context)

def play(context: GameContext, first: bool):
     play_vs_alphazero(
          context=context,
          first=first
     )

if __name__ == '__main__': # Needed for multiprocessing to work

     tic_tac_toe_path = "./models/test_nn"
     tic_tac_toe_context = GameContext(
          game_name="tic_tac_toe", 
          nn=NeuralNetwork().load(tic_tac_toe_path), 
          save_path=tic_tac_toe_path
     )

     connect4_path = "./models/connect_four/initial_test"
     connect4_context = GameContext(
          game_name="connect_four", 
          nn=NeuralNetworkConnectFour().load(connect4_path), 
          save_path=connect4_path
     )

     # test_overfit()
     # train_tic_tac_toe(tic_tac_toe_context)
     # train_connect_four(connect4_context)
     # self_play(context)
     play(tic_tac_toe_context, first=False)
     
     # create_tic_tac_toe_model("initial_test")
     # create_connect_four_model("initial_test")
