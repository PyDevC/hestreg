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
        self.cooldown = 0

    def detect(self, frame, ges, count=0):
        hist = []
        mean_hist = []
        eval_samples = 2
        num_classes = 27
        score_energy = torch.zeros((eval_samples, num_classes))
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
            out[-2:] = [0,0]
            hist.append(out)
            score_energy = torch.tensor(hist[-eval_samples:])
            curr_mean = torch.mean(score_energy, dim=0)
            mean_hist.append(curr_mean.cpu().numpy())
            value, indice = torch.topk(curr_mean, k=1)
            indices = np.argmax(out)
            self.pred = indices
            top_3 = out.argsort()[-3:]
            self.imgs = self.imgs[1:]
            if self.cooldown > 0:
                self.cooldown = self.cooldown - 1
            if value.item() > 0.5 and indices < 25 and self.cooldown == 0: 
                print('Gesture:', ges[indices], '\t\t\t\t\t\t Value: {:.2f}'.format(value.item()))
                self.cooldown = 16 
                return ges[indices]

        return count + 1

