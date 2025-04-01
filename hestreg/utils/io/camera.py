import cv2

def webcam(*args, **kwargs):
    """decorator for running functions in webcam
    """
    cam = cv2.VideoCapture(0)
    _, frame = cam.read()
    def wrapper(frame,*args, **kwargs):
        while frame:
            _, frame = cam.read()
            args[0](frame)
    return wrapper(frame, *args, **kwargs)


def release_cam(idx:cv2.VideoCapture):
    """releases the camera based on the index provided
    Parameters:
         idx: camera index   
    Return:
        release signal
    """
    if idx.isOpened():
        idx.release()
        return 0
    return 1
    
