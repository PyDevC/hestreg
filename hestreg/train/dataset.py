import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import functional
import cv2


class handGesture(Dataset):
    """
    Hand gesture dataset
    """
    def __init__(self, input_folder, resize):
        super().__init__()
        self.input_folder = input_folder
        self.resize = resize
        self.data, self.labels = self.load_data()

        
    def __getitem__(self, idx):
        image = self.data[idx]
        labels = self.data[idx]
        return image, labels

    def __len__(self):
        return len(self.data)

    def load_data(self):
        data = []
        labels = []
        gesture_folders = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
                   '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']

        for subject_id in range(10):
            subject_folder = f'0{subject_id}'
            for gesture_folder in gesture_folders:
                gesture_path = os.path.join(self.input_folder, subject_folder, gesture_folder)
                if not os.path.exists(gesture_path):
                    continue
                for img_file in os.listdir(gesture_path):
                    img_path = os.path.join(gesture_path, img_file)
                    img = cv2.imread(img_path)
                    img = functional.to_tensor(img)
                    img = functional.rgb_to_grayscale(img, num_output_channels=3)
                    img = functional.resize(img, size=[128,128])
                    data.append(img)
                    labels.append(gesture_folder)
    
        data = np.array(data)
        labels = np.array(labels)
        print(len(data))

        return data, labels

