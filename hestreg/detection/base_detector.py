class BaseDetector:
    def __init__(self, model):
        """base detector
        """
        self.model = model

    def detect(self, frame):
        pred = self.model.predict(frame)
        return pred
