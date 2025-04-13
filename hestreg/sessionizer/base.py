from hestreg.utils.model import ModelLoader

class BaseSession:
    """Base class for both window and window-less types of sessions
    """
    def __init__(self, name: str, *args, **kwargs):
        self.model = self.load_model()
        self.name = name 

    def load_model(self):
        loader = ModelLoader()
        model = loader.extract()
        return model

    def to(self, device):
        self.model.to(device)

    def start_session(self):
        pass

    def stop_session(self):
        pass
