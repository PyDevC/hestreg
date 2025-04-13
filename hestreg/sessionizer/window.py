from .base import BaseSession

class WindowSession(BaseSession):
    def __init__(self, name):
        super().__init__(name)
