import mouse

class mouseController:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.action = "" # checks the last performed action

    def move_to_hand(self, hold: bool=False):
        if hold:
            mouse.move(self.x, self.y)

    def right(self):
        mouse.right_click()

    def left(self):
        mouse.click()

    def get_function(self, handCoor, action):
        self.x = handCoor[0]
        self.y = handCoor[1]
        self._action_val(action)

    def _action_val(self, action):
        self.action = action
        if action == "Thumb Down":
            self.move_to_hand(hold=True)
