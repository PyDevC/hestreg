import torch.nn as nn

class base_cnn(nn.Module):
    def __init__(self, classes):
        self.classes = classes
        self.num_classes = len(classes)
    def forwared(self):
        pass

