import cv2

def open_window(frame, win_name):
    cv2.imshow(win_name, frame)
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        return 0
