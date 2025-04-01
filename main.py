import hestreg
from hestreg.sessionizer import window
from hestreg.detection import hand_detector
from hestreg.utils.io.window import open_window

model_name = "model"
win_opt = []

session = window.WindowSession(model_name, win_opt)

session.start_session()
while True:
    frame = session.export_frames()
    session.detector.detect(frame)
    if open_window(frame, 'window'):
        break

session.stop_session()
