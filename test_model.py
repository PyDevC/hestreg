import torch
import hestreg.utils.io.camera as camera
import os

path = os.getcwd()
model_path = os.path.join(path, "hestreg/models/handgest")

model = torch.load(model_path, weights_only=False)
model.eval()
model.to("cpu")

def digit_to_classname(digit):
    gesture_folders = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
                   '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']

    for idx, itm in enumerate(gesture_folders):
        if idx == digit.argmax():
            return itm


@camera.web_cam
def predict(frames, *args, **kwargs):
    frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
    frames.to("cpu")
    output = args[0](frames)
    print(f'{digit_to_classname(output[0].argmax())}')

predict(model)
