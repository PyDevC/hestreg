import cv2
from torchvision.transforms import ToTensor, functional

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

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (100,100))
    img = functional.to_tensor(img)
    return img

def web_cam(frame_trans):
    """decorator: around image processing functions
    """
    def wrapper(*args, **kwargs):
        cam = cv2.VideoCapture(0)
        sucess, frames = cam.read()
        while sucess:
            sucess, frames = cam.read()
            frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
            frames = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
            frames = cv2.resize(frames, (100,100))
            frames = functional.to_tensor(frames)
            frame_trans(frames,*args, **kwargs)
            if cv2.waitKey(1) == 'q':
                cv2.destroyAllWindows()
                break
    return wrapper
