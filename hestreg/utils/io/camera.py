import cv2

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
    cam = cv2.VideoCapture(0)
    sucess, frames = cam.read()
    while sucess:
        sucess, frames = cam.read()
        frame_trans(frames)
        if cv2.waitKey(1) == 'q':
            cv2.destroyAllWindows()
            break
