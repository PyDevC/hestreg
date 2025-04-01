import hestreg
from hestreg.sessionizer import window
from hestreg.detection import hand_detector
from hestreg.utils.io.window import open_window
from hestreg.utils.io.camera import webcam

model_name = "model"
win_opt = []

session = window.WindowSession(model_name, win_opt)

session.start_session()
while True:
    frame = session.export_frames()
    pred = session.detector.detect(frame)
    print(pred)
    print(2)
    set = open_window(frame, 'window')
    if set:
        break

session.stop_session()
