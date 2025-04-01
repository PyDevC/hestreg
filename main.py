import hestreg
from hestreg.sessionizer import window
from hestreg.detection import hand_detector

model_name = "model"
win_opt = []

session = window.WindowSession(model_name, win_opt)
session.start_session()

model = session.model
detector = hand_detector.HandDetector(model)
detector.detect()

session.close_session()
