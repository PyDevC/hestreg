import numpy as np
from .base_detector import BaseDetector

class HandDetector(BaseDetector):
    def __init__(self, model):
        super().__init__(model)

    def get_landmarks(self):
        """gets hand landmarks
        """
        lefthand = 0
        righthand = 0
        land_marks = [lefthand, righthand]
        return land_marks
