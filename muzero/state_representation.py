"""
1. value (vinner eller taper jeg i denne situasjonen?) -1 < x < 1 -> 0.93
2. action (hva skal jeg spille?) 
0,   1,   2
0.3, 0.1, 0.6
argmax(0.3, 0.1, 0.6) = 0.6

Nevralnett no 1. 
input 
[
    [0,  0,  0],
    [1, -1, -1],
    [-1, 1,  1]
]
output:
Hiden state (verdier som mennesker ikke kan forstå) (256/antall noder i hidden layer)
->


Value network
input -> 256 tall som ikke gir mening for mennesker (hidden layer)
output ->  -1 < x < 1


Policy network
input -> 256 tall som ikke gir mening for mennesker (hidden layer)
output -> 
0,   1,   2
0.3, 0.1, 0.6
argmax(0.3, 0.1, 0.6) = 0.6
...
xoo
oxx


"""

# valg 1:
# [
#     [0,  0,  0],
#     [1, -1, -1],
#     [-1, 1,  1]
# ]

# valg 2 (bedre):
# [
#     [
#         [0, 0, 0], # player 1
#         [0, 1, 1],
#         [1, 0, 0]
#     ],
#     [
#         [0, 0, 0], # player 2
#         [1, 0, 0],
#         [0, 1, 1]
#     ],
#     [
#         [1, 1, 1], # empty squares
#         [0, 0, 0],
#         [0, 0, 0]
#     ]
# ]

# Eksempel fra fire på rad:
"""
[[[0. 1. 0. 0. 1. 0. 0.],
  [0. 1. 0. 1. 0. 0. 0.],
  [0. 0. 0. 0. 0. 0. 0.],
  [0. 0. 0. 1. 0. 0. 0.],
  [0. 1. 0. 0. 0. 0. 0.],
  [0. 0. 0. 0. 0. 0. 0.]],

 [[0. 0. 0. 1. 0. 0. 1.],
  [0. 0. 0. 0. 0. 0. 0.],
  [0. 1. 0. 1. 0. 0. 0.],
  [0. 1. 0. 0. 0. 0. 0.],
  [0. 0. 0. 0. 0. 0. 0.],
  [0. 0. 0. 0. 0. 0. 0.]],

 [[1. 0. 1. 0. 0. 1. 0.],
  [1. 0. 1. 0. 1. 1. 1.],
  [1. 0. 1. 0. 1. 1. 1.],
  [1. 0. 1. 0. 1. 1. 1.],
  [1. 0. 1. 1. 1. 1. 1.],
  [1. 1. 1. 1. 1. 1. 1.]]]
"""

if __name__ == "__main__":
    from neural_network import NeuralNetwork
    import torch

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
    
    print(torch.__version__)
    print(state.size())

    x = nn.initial(state)
    print(x)
    print(state.size())
    print(x.size())
    print(torch.cuda.is_available()) # False if no GPU is available
