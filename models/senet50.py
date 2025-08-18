import torch.nn as nn

class SENet(nn.Module):
    def __init__(self,num_repeated_layers=50, use_squeeze_excitation=True):
        super().__init__()
        self.num_channels = 8
        self.conv_blocks = []
        pass


    def forward(self,x):
        pass

    def squeeze_excitation(self,X):
        # Squeeze step
        # Average Pool for Channel
        
        # Excitation Step (2 Fully Connected Layers)
        # Dimensionnality Reduction (r = 16)
        # Relu On Input Elements
        # Dimensionality Increasing
        # Softmax On the Results
        # Return result with wich we element wise multiply by the Input..
        # Added to the Input also
        pass

