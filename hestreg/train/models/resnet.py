import torch.nn as nn

def resnet(model, in_features):
    resnet = model
    in_features = resnet.fc.in_features
    fc = nn.Linear(in_features=in_features, out_features=10)
    resnet.fc = fc
    return resnet

