from src.neuralnet.neural_network import NeuralNetwork
import torch

def test_forward_propagation():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn = NeuralNetwork().to(device)
    state = torch.tensor(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 1]],  # player 1
            [[0, 0, 0], [0, 1, 1], [1, 0, 0]],  # player 2
            [[1, 1, 1], [0, 0, 0], [0, 0, 0]],  # empty squares
        ],
        dtype=torch.float,
    ).unsqueeze(0).to(device) # Unsqueeze to add batch dimension
    
    input_size: tuple[int] = state.size()
    output = nn.forward(state)

    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(output[0], torch.Tensor)
    assert isinstance(output[1], torch.Tensor)

print(torch.cuda.is_available()) # False if no GPU is available