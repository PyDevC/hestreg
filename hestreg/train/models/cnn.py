from . import _base
import torch.nn as nn

# add a custom type later
# type k_size = Union[T, tuple[T, T]]

class CNNModel(_base.base_cnn):
    r"""A basic CNN model
    Note: controlling first layer of CNN is important to handle data of 
    different formats.
    """
    def __init__(self, 
                 first_layer_channel: int,
                 kernel_size: tuple[int]
    )-> None:
        self.conv1 = nn.Conv2d(in_channels=first_layer_channel,
                               out_channels=64,
                               kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(64,128, kernel_size=(4,4)) # I want to be able to add more layers to the model by inhereiting this
    def forward(self):
        pass
        # I want to able to add many activation functions and other functionality
