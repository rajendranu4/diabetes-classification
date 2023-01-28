import torch.nn as nn


class DiabetesNetwork(nn.Module):
    def __init__(self, input_signals, hidden_neurons):
        super().__init__()

        # setting up and initializing the layers of neural network
        self.hidden_layer = nn.Linear(input_signals, hidden_neurons)
        self.output_layer = nn.Linear(hidden_neurons, 2)

        # activation functions for hidden and output layers
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, input_data):
        # feed forward to produce output
        x = self.hidden_layer(input_data)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.softmax(x)

        return x