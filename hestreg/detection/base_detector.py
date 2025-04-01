from torchvision.transforms import transforms, Compose 
import cv2
from PIL import Image
import torch

class BaseDetector:
    def __init__(self, model):
        """base detector
        """
        self.model = model

    def detect(self, frame):
        std, mean = [0.2674,  0.2676,  0.2648], [ 0.4377,  0.4047,  0.3925]
        transform = Compose([
            transforms.CenterCrop((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(std=std, mean=mean),
        ])

        frame = cv2.resize(frame, (160, 120))
        pre_img = Image.fromarray(frame.astype('uint8'), 'RGB')
        frame = transform(pre_img)
        data = torch.cat(frame).cuda()
        output = self.model(data.unsqueeze(0))
        out = (torch.nn.Softmax()(output).data).cpu().numpy()[0]
        return out
