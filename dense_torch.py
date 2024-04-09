import torch

def simple_training():
    """
    A stupdily simple example of training a neural network using PyTorch.
    The interesting part is that the training is very transparent, and you can see the weights and biases being updated.
    There are no optimizers, and the loss function is calculated manually as well.
    """
    
    W1 = torch.randn((3, 1), requires_grad=True)  # Weights for input to hidden layer
    b1 = torch.randn((3, 1), requires_grad=True)     # Biases for input to hidden layer

    W2 = torch.randn((1, 3), requires_grad=True)  # Weights for hidden to output layer
    b2 = torch.randn(1, requires_grad=True)     # Biases for hidden to output layer

    def relu(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, 0)

    def forward(x: torch.Tensor) -> torch.Tensor:
        hidden = relu(W1 @ x + b1)
        output = W2 @ hidden + b2
        return output.view(-1) # Remove the extra dimension

    learning_rate = 0.01
    epochs = 1000
    target = torch.tensor([5.0])

    for epoch in range(epochs):
        
        x = torch.tensor([[1.0]])        
        output = forward(x)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Output: {output.item()}')

        
        loss = (output - target).pow(2).mean() # Loss calculation (Mean Squared Error)
        
        loss.backward() # Backward pass (compute gradients)
        
    
        with torch.no_grad(): # No grad, since we don't want torch to believe that these operations somehow are part of the forward pass
            W1 -= learning_rate * W1.grad
            b1 -= learning_rate * b1.grad
            W2 -= learning_rate * W2.grad
            b2 -= learning_rate * b2.grad
            
            # Manually zero the gradients after updating weights
            W1.grad = None
            b1.grad = None
            W2.grad = None
            b2.grad = None 

    
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    test_output = forward(torch.tensor([[1.0]]))
    print(f'Output after training: {test_output.item()}')

def batch_training():

    W1 = torch.randn((3, 1), requires_grad=True)  # Weights for input to hidden layer
    b1 = torch.randn((3, 1), requires_grad=True)     # Biases for input to hidden layer

    W2 = torch.randn((1, 3), requires_grad=True)  # Weights for hidden to output layer
    b2 = torch.randn(1, requires_grad=True)     # Biases for hidden to output layer

    def relu(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, 0)

    def forward(x: torch.Tensor) -> torch.Tensor:
        hidden = relu(W1 @ x + b1)
        output = W2 @ hidden + b2
        return output.view(-1) # Remove the extra dimension

    learning_rate = 0.01
    num_accumulations = 10
    epochs = 10
    target = torch.tensor([5.0])

    for epoch in range(epochs):

        for i in range(num_accumulations):
            x = torch.tensor([[1.0]])        
            output = forward(x)
        
            loss = (output - target).pow(2).mean()
            loss.backward()
            if (epoch % 10 == 0 and i == 0):
                print(f'Epoch {epoch}, Output: {output.item()}')

        
        print(f'W1 grad: {W1.grad}, b1 grad: {b1.grad}, W2 grad: {W2.grad}, b2 grad: {b2.grad}')

        with torch.no_grad(): # No grad, since we don't want torch to believe that these operations somehow are part of the forward pass
            W1 -= learning_rate * W1.grad / num_accumulations
            b1 -= learning_rate * b1.grad / num_accumulations
            W2 -= learning_rate * W2.grad / num_accumulations
            b2 -= learning_rate * b2.grad / num_accumulations
            
            # Manually zero the gradients after updating weights
            W1.grad = None
            b1.grad = None
            W2.grad = None
            b2.grad = None 

    
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    test_output = forward(torch.tensor([[1.0]]))
    print(f'Output after training: {test_output.item()}')

"""
# Example of batch training with gradient accumulation, using optimizer

optimizer = torch.optim.SGD([W1, b1, W2, b2], lr=learning_rate)

for epoch in range(epochs):
    for i in range(num_accumulations):
        x = torch.tensor([[1.0]])        
        output = forward(x)
        loss = (output - target).pow(2).mean() 
        loss.backward()
    
    # Scale down gradients by the number of accumulations
    for param in [W1, b1, W2, b2]:
        if param.grad is not None:
            param.grad /= num_accumulations
    
    optimizer.step()  # Apply gradients
    optimizer.zero_grad()  # Reset gradients for the next set of accumulations
"""

# simple_training()
# batch_training()

import torch.nn as nn
from icecream import ic

# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()

# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
ic(input)
for _ in range(10000):
    output = loss(input, target)
    output.backward()
    with torch.no_grad():
        input -= 0.1 * input.grad
        input.grad.zero_()
    print(f'Loss: {output.item()}')
    ic(input)

ic(input)
ic(target)
ic(input.softmax(dim=1))


