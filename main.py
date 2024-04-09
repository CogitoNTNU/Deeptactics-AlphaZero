from src.alphazero.alphazero import play_alphazero
from src.alphazero.alphazero_training import train_alphazero
import torch

# play_alphazero()

# train_alphazero(10, 100)

for _ in range(50):
     train_alphazero(20, 60)


