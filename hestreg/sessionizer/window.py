from .base import BaseSession
from utils import opterations as opt
from utils.io import camera, window

class WindowSession(BaseSession):
    def __init__(self, model_name, window_opt:list[str]):
        super().__init__(model_name)

        # apply all the window options in window_opt

    @camera.webcam
    def screening(
        self,
        frame,
        *args,
        **kwargs
    ):
        """Apply preprocessing methods on image using this

        Parameters:
            frame: PIL image from video or webcam
            resize: adjust the quality of video input (lower the faster but
            worsen the quality of predcition)

        Returns: preprocessed frame
        """
        return frame

    def start_session(self):
        self.screening()
