import cv2
from hestreg.sessionizer import BaseSession
from hestreg.detection import BaseDetector

predictions = [
    "Swiping Left",
    "Swiping Right",
    "Swiping Down",
    "Swiping Up",
    "Pushing Hand Away",
    "Pulling Hand In",
    "Sliding Two Fingers Left",
    "Sliding Two Fingers Right",
    "Sliding Two Fingers Down",
    "Sliding Two Fingers Up",
    "Pushing Two Fingers Away",
    "Pulling Two Fingers In",
    "Rolling Hand Forward",
    "Rolling Hand Backward",
    "Turning Hand Clockwise",
    "Turning Hand Counterclockwise",
    "Zooming In With Full Hand",
    "Zooming Out With Full Hand",
    "Zooming In With Two Fingers",
    "Zooming Out With Two Fingers",
    "Thumb Up",
    "Thumb Down",
    "Shaking Hand",
    "Stop Sign",
    "Drumming Fingers",
    "No gesture",
    "Doing other things",
]

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 64)
success, frame = camera.read()

session = BaseSession('session')
detector = BaseDetector(session.model)
# start session

count = 0
while success:
    success, frame = camera.read()

    out = detector.detect(frame, count)
    if isinstance(out, int):
        count = out
    else:
        print(predictions[detector.pred])

# stop session
camera.release()
cv2.destroyAllWindows()
