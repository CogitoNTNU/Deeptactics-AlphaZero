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


def test_overfit(context: GameContext):
     
     mp.set_start_method('spawn')
     train_alphazero_model(
          context=context,
          num_games=3,
          num_simulations=100,
          epochs=1,
          batch_size=64
     )

def train_tic_tac_toe(context: GameContext):
     mp.set_start_method('spawn')

     try:
          for i in range(int(1e6)):
               train_alphazero_model(
                    context=context,
                    num_games=48,
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
                    num_games=20,
                    num_simulations=200,
                    epochs=3,
                    batch_size=256,
               )
               print(f'Training session {i + 1} finished!')
     except KeyboardInterrupt:
          print('Training interrupted!')


def self_play(context: GameContext):
     alphazero_self_play(context)

def play(context: GameContext, first: bool, mcts: bool = False):
     play_vs_alphazero(
          context=context,
          first=first,
          mcts=mcts
     )

overfit_path = "./models/connect_four/overfit_nn"
overfit_context = GameContext(
     game_name="connect_four", 
     nn=NeuralNetworkConnectFour().load(overfit_path), 
     save_path="./models/overfit_waste"
)

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


if __name__ == '__main__': # Needed for multiprocessing to work

     

     # test_overfit(overfit_context)
     # train_tic_tac_toe(tic_tac_toe_context)
     # train_connect_four(connect4_context)
     # self_play(tic_tac_toe_context)
     # self_play(connect4_context)
     # play(tic_tac_toe_context, first=False)
     play(connect4_context, first=False)
     # play(connect4_context, first=True, mcts=True)

     # create_tic_tac_toe_model("initial_test")
     # create_connect_four_model("overfit_nn")
