from hestreg.utils.io import camera, window
from hestreg.detection.hand_detector import HandDetector
from hestreg.utils.model import ModelLoader

# apply type hints to the classes and methods

class BaseSession:
    """Base class for both window and window-less types of sessions
    """
    def __init__(self, model_name:str, *args, **kwargs):
        self.model = self.load_model(model_name)

        # check if the args contains any accelerator name
        """
        arg_devices = [device for device in available_devices() if device in args]

        if arg_devices:
            self.accelerator = arg_devices[0]
        else:
            self.accelerator = get_device()
        """
        self.detector = HandDetector(self.model)


    def load_model(self, model_name:str):
        loader = ModelLoader()
        model = loader.extract(model_name)
        return model

    def name(self):
        """return the name of the session
        i.e., name of the session should be same as name of the session
        """
        return self.__class__.__name__
