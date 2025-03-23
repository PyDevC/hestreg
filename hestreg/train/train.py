import torch
import numpy as np
from torch.utils.data import DataLoader

def train_model(model, dataset, criterian, optimizer, device):
    batch = 24
    num_epoch = 10

    labels = dataset.labels

    train_dataset = [(torch.tensor(dataset.data[idx], dtype=torch.float32).unsqueeze(0), torch.tensor(np.long(labels[idx][0]), dtype=torch.long)) for idx in range(len(dataset))]

    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    model.to(device)

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for input, label in train_dataloader:
            input = input.to(device)
            print(input.shape)

            label = label.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterian(output, label)
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch[{epoch}/{num_epoch}]") # change it to verbose or any other method for better control over stdout
    return model
