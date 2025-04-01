import cv2

def webcam(screening):
    """decorator for running functions in webcam
    """
    #check the type of function input

    def wrapper(*args, **kwargs):
        cam = cv2.VideoCapture(0)
        success, frame = cam.read()
        while success:
            success, frame = cam.read()
            frame = screening(frame,*args, **kwargs)
        return frame
    return wrapper


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
    
