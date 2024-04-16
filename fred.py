import torch
import pyspiel
from src.neuralnet.neural_network import NeuralNetwork
from icecream import ic

nn = NeuralNetwork.load('models/test_nn')

ic(nn.forward)