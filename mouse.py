from pynput.mouse import Button, Controller

class RemoteMouse:
    def __init__(self):
        self.mouse = Controller()

    def getPosition(self):
        return self.mouse.position

    def setPos(self, xPos, yPos):
        self.mouse.position = (xPos, yPos)

    def movePos(self, xPos, yPos):
        self.mouse.move(xPos, yPos)

    def click(self):
        self.mouse.click(Button.left)

    def doubleClick(self):
        self.mouse.click(Button.left, 2)

    def clickRight(self):
        self.mouse.click(Button.right)

    def drag(self, from_x, from_y, to_x, to_y, is_absolute=True):
        if is_absolute is True:
            self.mouse.position = (from_x, from_y)
        else:
            self.mouse.position = self.getPosition()
            self.click()
            self.mouse.move(from_x, from_y)
        self.click()
        self.mouse.press(Button.left)

        if is_absolute is True:
            self.mouse.position = (to_x, to_y)
        else:
            self.mouse.move(to_x, to_y)
        self.mouse.release(Button.left)

if __name__ == '__main__':
    mouse = RemoteMouse()
    print('X: %s, Y:%s' %mouse.getPosition())
    
    # mouse.setPos(200, 200)
    # mouse.movePos(400, 400)
    # mouse.click()
    mouse.doubleClick()
    # mouse.drag(200, 200, 400, 400)