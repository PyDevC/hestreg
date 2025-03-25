import hestreg.utils.io.camera as camera

@camera.web_cam
def camera_working():
    return "camera working"
