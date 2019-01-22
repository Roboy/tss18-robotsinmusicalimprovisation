import sys
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget, QDial, QPushButton, QGroupBox, QGridLayout



class Poti(QWidget):
    def __init__(self, mean, variance, parent=None):
        super(Poti, self).__init__(parent=parent)
        # self.verticalLayout = QVBoxLayout()
        # self.horizontalLayout = QHBoxLayout()
        self.label = QLabel(self)
        self.dial = QDial(self)
        # self.horizontalLayout.addWidget(self.dial)
        # self.horizontalLayout.addWidget(self.label)
        # self.verticalLayout.addLayout(self.horizontalLayout)

        self.current_value = None
        self.dial.valueChanged.connect(self.setLabelValue)
        self.mean = mean
        self.variance = variance
        self.dial.setMinimum(0)
        self.dial.setMaximum(200)
        self.minimum = mean - variance
        self.maximum = mean + variance
        self.setLabelValue(self.dial.value())
        self.dial.setSliderPosition(100)
        # print("max", self.dial.maximum())

    def setLabelValue(self, value):
        self.current_value = self.minimum + (float(value) /
            (self.dial.maximum() - self.dial.minimum())) * (self.maximum - self.minimum)
        self.label.setText("%.2f" % self.current_value)

    def updatePoti(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def reset(self):
        self.dial.setSliderPosition(100)

    def randomPos(self):
        rand = np.random.randint(0,200)
        self.dial.setSliderPosition(rand)



class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent=parent)
        self.title = 'VAEsemane'
        self.left = 20
        self.top = 20
        self.width = 320
        self.height = 100
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.b_reset = QPushButton('Reset')
        self.b_random = QPushButton('Randomize')
        self.potis = []
        for i in range(100):
            self.potis.append(Poti(0.,10.))

        self.b_reset.clicked.connect(self.reset_click)
        self.b_random.clicked.connect(self.random_click)
        self.createGridLayout()
        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.horizontalGroupBox)
        windowLayout.addWidget(self.b_reset)
        windowLayout.addWidget(self.b_random)
        self.setLayout(windowLayout)

    def createGridLayout(self):
        self.horizontalGroupBox = QGroupBox()
        layout = QGridLayout()
        for i in range(10):
            for j in range(10):
                layout.addWidget(self.potis[i*10+j],i,j)
        self.horizontalGroupBox.setLayout(layout)

    def reset_click(self):
        for poti in self.potis:
            poti.reset()

    def random_click(self):
        for poti in self.potis:
            poti.randomPos()


    def update(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())
