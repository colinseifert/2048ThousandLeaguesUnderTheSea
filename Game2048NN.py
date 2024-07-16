import torch
import torch.nn

class Game2048NN(torch.nn.Module):
    def __init__(self):
        super(Game2048NN, self).__init__()
        # Define the input layer (16 inputs) and the first hidden layer
        self.fc1 = torch.nn.Linear(16, 4)  # Output layer with 4 neurons (one for each move)

    def forward(self, x):
        # Pass the input through the layers
        x = self.fc1(x)  # No activation function here as we will apply it during loss computation
        return x