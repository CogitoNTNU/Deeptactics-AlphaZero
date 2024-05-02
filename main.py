from argparse import ArgumentParser, ArgumentTypeError
from torch import multiprocessing as mp
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
          epochs=1000,
          batch_size=256
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
                    num_simulations=100,
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

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

parser: ArgumentParser = ArgumentParser(description='Control the execution of the AlphaZero game playing system.')
parser.add_argument('--test_overfit', action='store_true', help='Test overfitting on Connect Four.')
parser.add_argument('--train_tic_tac_toe', action='store_true', help='Train AlphaZero on Tic Tac Toe.')
parser.add_argument('--train_connect_four', action='store_true', help='Train AlphaZero on Connect Four for a long time.')

parser.add_argument('--self_play_ttt', action='store_true', help='Run self-play on Tic Tac Toe.')
parser.add_argument('--self_play_c4', action='store_true', help='Run self-play on Connect Four.')

parser.add_argument('--play_ttt', action='store_true', help='Enable playing against AlphaZero on Tic Tac Toe.')
parser.add_argument('--play_c4', action='store_true', help='Play against AlphaZero on Connect Four.')

parser.add_argument('-f', '--first', action='store_true', help='Play first in the game.')
parser.add_argument('-m', '--mcts', action='store_true', help='Replace human player with MCTS.')

args = parser.parse_args()



if __name__ == '__main__': # Needed for multiprocessing to work

     if args.test_overfit:
          test_overfit(overfit_context)
     
     if args.train_tic_tac_toe:
          train_tic_tac_toe(tic_tac_toe_context)

     if args.train_connect_four:
          train_connect_four(connect4_context)

     if args.self_play_ttt:
          self_play(tic_tac_toe_context)
     
     if args.self_play_c4:
          self_play(connect4_context)

     if args.play_ttt:
          play(tic_tac_toe_context, first=args.first)
     
     if args.play_c4:
          play(connect4_context, first=args.first, mcts=args.mcts)

     
     # create_tic_tac_toe_model("initial_test")
     # create_connect_four_model("overfit_nn")
