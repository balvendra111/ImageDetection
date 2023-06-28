# from fastapi import FastAPI
# from PySide6.QtWidgets import QApplication
# from PySide6.QtGui import QPixmap
# from PySide6.QtCore import QFile, Signal
# from PySide6.QtUiTools import QUiLoader
# from PySide6.QtCore import QThread
# from PySide6.QtCore import QDir, Qt
# from PySide6.QtWidgets import QWidget
# from PySide6.QtWidgets import QFileDialog
# from PySide6.QtGui import QImage
# from fastapi import FastAPI, UploadFile, File
# from PySide6.QtWidgets import QApplication, QWidget, QFileDialog
# from PySide6.QtCore import QFile, Signal
# from PySide6.QtUiTools import QUiLoader
# from PySide6.QtGui import QPixmap, QImage
# from PySide6.QtCore import QThread, Qt
# import cv2
# import numpy as np
# import base64
# import uvicorn
# from fastapi import FastAPI
# from PySide6.QtWidgets import QApplication
# from PySide6.QtGui import QPixmap
# from PySide6.QtCore import QFile, Signal
# from PySide6.QtUiTools import QUiLoader
# from PySide6.QtCore import QThread
# from PySide6.QtCore import QDir, Qt
# from PySide6.QtWidgets import QWidget
# from PySide6.QtWidgets import QFileDialog
# from PySide6.QtGui import QImage
# import sys
# import os
# import io
# import cv2
# import numpy as np
# import base64
# from PIL import Image
# from detector import Detector


# from fastapi import FastAPI, UploadFile, File
# from PySide6.QtWidgets import QApplication, QWidget, QFileDialog
# from PySide6.QtCore import QFile, Signal
# from PySide6.QtUiTools import QUiLoader
# from PySide6.QtGui import QPixmap, QImage
# from PySide6.QtCore import QThread

# app = FastAPI()

# class ProcessImage(QThread):
#     signal_show_frame = Signal(object)

#     def __init__(self, fileName):
#         QThread.__init__(self)
#         self.fileName = fileName

#         from detector import Detector
#         self.detector = Detector()

#     def run(self):
#         self.video = cv2.VideoCapture(self.fileName)
#         while True:
#             valid, self.frame = self.video.read()
#             if valid is not True:
#                 break
#             self.frame = self.detector.detect(self.frame)
#             self.signal_show_frame.emit(self.frame)
#             cv2.waitKey(30)
#         self.video.release()

#     def stop(self):
#         try:
#             self.video.release()
#         except:
#             pass


# class show(QThread):
#     signal_show_image = Signal(object)

#     def __init__(self, fileName):
#         QThread.__init__(self)
#         self.fileName = fileName
#         self.video = cv2.VideoCapture(self.fileName)

#     def run(self):
#         while True:
#             valid, self.frame = self.video.read()
#             if valid is not True:
#                 break
#             self.signal_show_image.emit(self.frame)
#             cv2.waitKey(30)
#         self.video.release()

#     def stop(self):
#         try:
#             self.video.release()
#         except:
#             pass


# class MainWindow(QWidget):
#     def __init__(self):
#         super(MainWindow, self).__init__()
#         loader = QUiLoader()
#         self.ui = loader.load("ui/form.ui")

#         self.ui.btn_browse.clicked.connect(self.getFile)
#         self.ui.btn_start.clicked.connect(self.predict)

#         self.ui.show()

#     def getFile(self):
#         self.fileName = QFileDialog.getOpenFileName(
#             self, 'Single File', 'C:\'',
#             '*.jpg *.mp4 *.jpeg *.png *.avi'
#         )[0]
#         self.ui.txt_address.setText(str(self.fileName))
#         self.show = show(self.fileName)
#         self.show.signal_show_image.connect(self.show_input)
#         self.show.start()

#     def predict(self):
#         self.process_image = ProcessImage(self.fileName)
#         self.process_image.signal_show_frame.connect(self.show_output)
#         self.process_image.start()

#     def show_input(self, image):
#         pixmap = convertCVImage2QtImage(image)
#         self.ui.lbl_input.setPixmap(pixmap)

#     def show_output(self, image):
#         pixmap = convertCVImage2QtImage(image)
#         self.ui.lbl_output.setPixmap(pixmap)


# def convertCVImage2QtImage(cv_img):
#     cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#     height, width, channel = cv_img.shape
#     bytesPerLine = 3 * width
#     qimg = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
#     return QPixmap.fromImage(qimg)


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.post("/process-image")
# async def process_image(file: UploadFile = File(...)):
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     detector = Detector()
#     processed_image = detector.detect(img)

#     # Convert processed image to PIL Image
#     processed_pil_image = Image.fromarray(processed_image)

#     # Create an in-memory stream to store the image
#     img_io = io.BytesIO()
#     processed_pil_image.save(img_io, format='JPEG')
#     img_io.seek(0)

#     # Return the image as a streaming response
#     return StreamingResponse(img_io, media_type="image/jpeg")


# def run_gui():
#     app = QApplication([])
#     widget = MainWindow()
#     app.exec()


# if __name__ == "__main__":
#     run_gui() 
# if __name__ == "__main__":   
#     uvicorn.run(app, host="127.0.0.1", port=5555)

from fastapi import FastAPI, UploadFile, File
from PySide6.QtWidgets import QApplication, QWidget, QFileDialog
from PySide6.QtCore import Signal
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QThread
import uvicorn
from detector import Detector
import cv2
import numpy as np
import io
from PIL import Image
from starlette.responses import StreamingResponse

app = FastAPI()


class ProcessImage(QThread):
    signal_show_frame = Signal(object)

    def __init__(self, fileName):
        QThread.__init__(self)
        self.fileName = fileName

        from detector import Detector
        self.detector = Detector()

    def run(self):
        self.video = cv2.VideoCapture(self.fileName)
        while True:
            valid, self.frame = self.video.read()
            if valid is not True:
                break
            self.frame = self.detector.detect(self.frame)
            self.signal_show_frame.emit(self.frame)
            cv2.waitKey(30)
        self.video.release()

    def stop(self):
        try:
            self.video.release()
        except:
            pass


class Show(QThread):
    signal_show_image = Signal(object)

    def __init__(self, fileName):
        QThread.__init__(self)
        self.fileName = fileName
        self.video = cv2.VideoCapture(self.fileName)

    def run(self):
        while True:
            valid, self.frame = self.video.read()
            if valid is not True:
                break
            self.signal_show_image.emit(self.frame)
            cv2.waitKey(30)
        self.video.release()

    def stop(self):
        try:
            self.video.release()
        except:
            pass


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        loader = QUiLoader()
        self.ui = loader.load("ui/form.ui")

        self.ui.btn_browse.clicked.connect(self.getFile)
        self.ui.btn_start.clicked.connect(self.predict)

        self.ui.show()

    def getFile(self):
        self.fileName = QFileDialog.getOpenFileName(
            self, 'Single File', 'C:\'',
            '*.jpg *.mp4 *.jpeg *.png *.avi'
        )[0]
        self.ui.txt_address.setText(str(self.fileName))
        self.show = Show(self.fileName)
        self.show.signal_show_image.connect(self.show_input)
        self.show.start()

    def predict(self):
        self.process_image = ProcessImage(self.fileName)
        self.process_image.signal_show_frame.connect(self.show_output)
        self.process_image.start()

    def show_input(self, image):
        pixmap = convertCVImage2QtImage(image)
        self.ui.lbl_input.setPixmap(pixmap)

    def show_output(self, image):
        pixmap = convertCVImage2QtImage(image)
        self.ui.lbl_output.setPixmap(pixmap)


# def convertCVImage2QtImage(cv_img):
#     if len(cv_img.shape) == 2:  # Grayscale image
#         cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
#     elif len(cv_img.shape) == 3:  # Color image
#         cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#     height, width, channel = cv_img.shape
#     bytesPerLine = 3 * width
#     qimg = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
#     return QPixmap.fromImage(qimg)
def convertCVImage2QtImage(cv_img):
    if isinstance(cv_img, tuple):  # Check if image is a tuple
        cv_img = cv_img[0]  # Extract the actual image from the tuple
    if len(cv_img.shape) == 2:  # Grayscale image
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    elif len(cv_img.shape) == 3:  # Color image
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    height, width, channel = cv_img.shape
    bytesPerLine = 3 * width
    qimg = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    detector = Detector()
    processed_image = detector.detect(img)

    if isinstance(processed_image, tuple):  # Check if processed_image is a tuple
        processed_image = processed_image[0]  # Extract the actual image from the tuple

    # Convert processed image to PIL Image
    processed_pil_image = Image.fromarray(processed_image)

    # Create an in-memory stream to store the image
    img_io = io.BytesIO()
    processed_pil_image.save(img_io, format='JPEG')
    img_io.seek(0)

    # Return the image as a streaming response
    return StreamingResponse(img_io, media_type="image/jpeg")


def run_gui():
    app = QApplication([])
    widget = MainWindow()
    app.exec()


if __name__ == "__main__":
    run_gui()
    
if __name__ == "__main__":   
    uvicorn.run(app, host="127.0.0.1", port=5555)
