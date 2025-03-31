from accelerator import get_device, available_devices

class BaseSession:
    """Base class for both window and window less types of sessions
    """
    def __init__(self, model_name:str, session_type:str, *args, **kwargs):
        self.model = self.load_model(model_name)

        # check if the args contains any accelerator name
        arg_devices = [device for device in available_devices() if device in args]

        if arg_devices:
            self.accelerator = arg_devices[0]
        else:
            self.accelerator = get_device()

    def load_model(self, model_name:str):
        pass # no model architecture is decided so we will decide there

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
        pass
