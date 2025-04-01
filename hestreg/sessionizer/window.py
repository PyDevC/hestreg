from .base import BaseSession
from utils import opterations as opt
from utils import io

class WindowSession(BaseSession):
    def __init__(self, model_name, session_type, window_opt:list[str]):
        super().__init__(model_name, session_type)

        # apply all the window options in window_opt

    def screening(
        self,
        frame,
        resize,
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
        frame = opt.resize(frame, resize)
        frame = opt.grayscale(frame)

        # apply all the opterations mentioned in args and kwargs

        return frame
