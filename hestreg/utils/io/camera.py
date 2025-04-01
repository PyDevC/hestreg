import cv2

def webcam(screening,*args, **kwargs):
    """decorator for running functions in webcam
    """
    #check the type of function input

    cam = cv2.VideoCapture(0)
    _, frame = cam.read()
    def wrapper(frame,*args, **kwargs):
        while frame:
            _, frame = cam.read()
            frame = screening(frame)
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
    
