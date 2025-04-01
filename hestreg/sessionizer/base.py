from utils.io import camera, window
from detection.hand_detector import HandDetector
#from model import extract

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
        pass # no model architecture is decided so we will decide there

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

    def name(self):
        """return the name of the session
        i.e., name of the session should be same as name of the session
        """
        return self.__class__.__name__
