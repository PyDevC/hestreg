from .base import BaseSession
from hestreg.utils.io import camera, window

class WindowSession(BaseSession):
    def __init__(self, model_name, window_opt:list[str]):
        super().__init__(model_name)

        # apply all the window options in window_opt

    def start_session(self, *args):
        self.frame = self.screening(0)

    def stop_session(self):
        exit(0)

    def export_frames(self):
        return self.frame

