import torch
import hestreg.utils.io.camera as camera
import os

path = os.getcwd()
model_path = os.path.join(path, "hestreg/models/handgest")

model = torch.load(model_path, weights_only=False)
model.eval()
input_folder = os.path.join(path, "data/leapgestrecog/leapGestRecog/00/01_palm/frame_00_01_0001.png")

def digit_to_classname(digit):
    gesture_folders = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
                   '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']

    for idx, itm in enumerate(gesture_folders):
        if idx == digit:
            return itm

image = camera.load_image(input_folder)
model = torch.load(model_path, weights_only=False)
model.eval()
print(model(input))
