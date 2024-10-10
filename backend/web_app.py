from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from torch import multiprocessing as mp
from src.neuralnet.neural_network import NeuralNetwork
from src.neuralnet.neural_network_connect_four import NeuralNetworkConnectFour
from src.alphazero.agents.alphazero_play_agent import alphazero_self_play
from src.alphazero.alphazero_train_model import train_alphazero_model
from src.play.play_vs_alphazero import main as play_vs_alphazero
from src.utils.game_context import GameContext

app = FastAPI()

# Predefined game contexts
overfit_path = "./models/overfit/connect4_nn.nn"
overfit_context = GameContext(
    game_name="connect_four", 
    nn=NeuralNetworkConnectFour().load(overfit_path), 
    save_path="./models/overfit/connect4_overfit_waste.nn"
)

tic_tac_toe_path = "./models/tic_tac_toe/good_nn.nn"
tic_tac_toe_context = GameContext(
    game_name="tic_tac_toe", 
    nn=NeuralNetwork().load(tic_tac_toe_path), 
    save_path=tic_tac_toe_path
)

connect4_path = "./models/connect_four/good_nn.nn"
connect4_context = GameContext(
    game_name="connect_four", 
    nn=NeuralNetworkConnectFour().load(connect4_path), 
    save_path=connect4_path
)

# Background task wrappers
def train_tic_tac_toe_task(context: GameContext):
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

def train_connect_four_task(context: GameContext):
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

@app.post("/test_overfit")
async def test_overfit():
    try:
        mp.set_start_method('spawn')
        train_alphazero_model(
            context=overfit_context,
            num_games=3,
            num_simulations=100,
            epochs=1000,
            batch_size=256
        )
        return {"status": "Test overfit completed!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_ttt")
async def train_tic_tac_toe(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_tic_tac_toe_task, tic_tac_toe_context)
    return {"status": "Training Tic Tac Toe model started in background."}

@app.post("/train_c4")
async def train_connect_four(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_connect_four_task, connect4_context)
    return {"status": "Training Connect Four model started in background."}

class PlayRequest(BaseModel):
    game: str
    first: bool
    mcts: Optional[bool] = False

@app.post("/self_play")
async def self_play(game: str):
    context = tic_tac_toe_context if game == "tic_tac_toe" else connect4_context
    try:
        alphazero_self_play(context)
        return {"status": f"Self-play for {game} completed!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/play")
async def play_game(play_request: PlayRequest):
    game = play_request.game
    context = tic_tac_toe_context if game == "tic_tac_toe" else connect4_context
    try:
        play_vs_alphazero(context=context, first=play_request.first, mcts=play_request.mcts)
        return {"status": f"Playing against AlphaZero on {game} completed!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
