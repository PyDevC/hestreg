import cv2
from ..preprocess.color import denoising, grayscale, blur

def load_frames(vid):
    """Loads the frames from a video to an array for processing
    """
    cam = cv2.VideoCapture(vid)
    images = []
    sucess, frames = cam.read()
    while sucess:
        sucess, frames = cam.read() # check what will be returned by this statement and if not campatible as an array then convert it into an array
        images.append(frames)
    return images


def web_cam(frame_trans):
    """decorator: around image processing functions
    """
    def wrapper(*args, **kwargs):
        cam = cv2.VideoCapture(0)
        sucess, frames = cam.read()
        while sucess:
            sucess, frames = cam.read()
            frames = cv2.resize(frames, (100,100))
            print(type(frames))
            frames = grayscale(frames)
            frames = blur(frames)
            frames = denoising(frames)
            
            frame_trans(frames,*args, **kwargs)
            if cv2.waitKey(1) == 'q':
                cv2.destroyAllWindows()
                break
    return wrapper
