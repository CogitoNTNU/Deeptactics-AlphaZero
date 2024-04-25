from src.neuralnet.neural_network import NeuralNetwork
from src.neuralnet.neural_network_connect_four import NeuralNetworkConnectFour

def create_tic_tac_toe_model(model_name: str):
    nn = NeuralNetwork()
    nn.save(f"./models/tic_tac_toe/{model_name}")

def create_connect_four_model(model_name: str):
    nn = NeuralNetworkConnectFour()
    nn.save(f"./models/connect_four/{model_name}")