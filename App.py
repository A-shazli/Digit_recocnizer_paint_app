from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget
from keras.models import load_model
import sys
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import PyQt5 as Qt
import tensorflow as tf
from PIL import ImageQt
model = load_model('mnist.h5')

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.label = QtWidgets.QLabel()
        self.label_pred = QtWidgets.QLabel()
        self.label_pred.setText("Draw a number!")
        self.label_pred.setFont(QFont('Times', 10))
        btn = QPushButton("Recognize", self)
        btn_c = QPushButton("Clear", self)
        self.canvas = QtGui.QPixmap(400, 300)
        self.label.setPixmap(self.canvas)
        self.label.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.label_pred)
        self.layout.addWidget(btn)
        self.layout.addWidget(btn_c)
        btn.clicked.connect(self.save_img)
        btn_c.clicked.connect(self.clear)
        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        self.last_x, self.last_y = None, None




    def predict(self, img):
        img = img.resize((28, 28))
        plt.imshow(img)
        plt.title("Input image after resizing", fontsize= 20)
        plt.show()
        img = img.convert('L')
        img = np.array(img)
        img = img.reshape(1, 28, 28, 1)
        img = img / 255.0
        pred = model.predict(img)[0]
        pred = tf.nn.softmax(pred)
        pred = pred.numpy() * 100
        self.label_pred.setText("The number is: " + str(np.argmax(pred)) + ", " + str(max(pred)) + "% sure")





    def mouseMoveEvent(self, e):
        if self.last_x is None:  # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return  # Ignore the first time.

        self.painter = QtGui.QPainter(self.label.pixmap())
        pen = QtGui.QPen()
        pen.setWidth(15)
        pen.setColor(QtGui.QColor("white"))
        self.painter.setPen(pen)
        self.painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        self.painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    # func to forget the last location when mouse is released and be able to draw form new loc
    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def save_img(self):
        image = ImageQt.fromqpixmap(self.label.pixmap())
        # image.save('test.png')

        self.predict(image)


    def clear(self):

        self.label.setPixmap(self.canvas)
        self.label_pred.setText("Draw a number!")





app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()