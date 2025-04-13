from torchvision.transforms import transforms, Compose 
import cv2
import torch
from PIL import Image
import numpy as np

class BaseDetector:
    def __init__(self, model):
        """base detector
        """
        self.model = model
        self.imgs = []
        self.pred = 0

    def detect(self, frame, count=0):
        #count = 0 # counts the number of frames that have passed
        std, mean = [0.2674,  0.2676,  0.2648], [ 0.4377,  0.4047,  0.3925]
        transform = Compose([
            transforms.CenterCrop((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(std=std, mean=mean),
        ])

        frame = cv2.resize(frame, (160, 120))
        pre_img = Image.fromarray(frame.astype('uint8'), 'RGB')

        img = transform(pre_img)

        if count%4 == 0:
            self.imgs.append(torch.unsqueeze(img, 0))

        if len(self.imgs) == 16:
            data = torch.cat(self.imgs).cuda()
            output = self.model(data.unsqueeze(0))
            out = (torch.nn.Softmax()(output).data).cpu().numpy()[0]
            self.pred = np.argmax(out)
            top_3 = out.argsort()[-3:]
            self.imgs = self.imgs[3:]
            return top_3


        if len(self.imgs) >= 17:
            self.imgs = self.imgs[4:]

        return count + 1

    def _check_threshold(self, out):
        self.threshold = 0.5
        if max(out) > self.threshold:
            return True
        return False
