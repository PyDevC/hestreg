from hestreg.train.train import train_model
from hestreg.train.dataset import handGesture
from hestreg.train.models.cnn import CNNModel
from hestreg.train.loss import cross_entropy_loss
import torch.optim as optim
import torch
import os


path = os.getcwd()
input_folder = os.path.join(path, "data/leapgestrecog/leapGestRecog")
save_path = os.path.join(path, "hestreg/models/handgest")

gesture_folders = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
                   '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']

device = "cuda"
model = CNNModel(gesture_folders)
criterian = cross_entropy_loss
optimzer = optim.Adam(model.parameters(), lr=0.01)

dataset = handGesture(input_folder=input_folder, resize=(100,100))

model = train_model(model, dataset, criterian, optimzer, device)
torch.save(model, save_path)
