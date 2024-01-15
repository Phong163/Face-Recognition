# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(614, 427)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(30, 40, 561, 311))
        self.stackedWidget.setAutoFillBackground(False)
        self.stackedWidget.setStyleSheet("")
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_1 = QtWidgets.QWidget()
        self.page_1.setStyleSheet("background-color: rgb(85, 170, 127);")
        self.page_1.setObjectName("page_1")
        self.label = QtWidgets.QLabel(self.page_1)
        self.label.setGeometry(QtCore.QRect(-10, 0, 571, 331))
        self.label.setText("")
        self.label.setObjectName("label")
        self.stackedWidget.addWidget(self.page_1)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setStyleSheet("background-color: rgb(255, 170, 127);")
        self.page_2.setObjectName("page_2")
        self.label_2 = QtWidgets.QLabel(self.page_2)
        self.label_2.setGeometry(QtCore.QRect(0, 0, 561, 311))
        self.label_2.setObjectName("label_2")
        self.stackedWidget.addWidget(self.page_2)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.page_3.setObjectName("page_3")
        self.label_3 = QtWidgets.QLabel(self.page_3)
        self.label_3.setGeometry(QtCore.QRect(30, 50, 61, 41))
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("Picture3.png"))
        self.label_3.setScaledContents(True)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.page_3)
        self.label_4.setGeometry(QtCore.QRect(40, 150, 51, 41))
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap("Picture2.jpg"))
        self.label_4.setScaledContents(True)
        self.label_4.setWordWrap(False)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.page_3)
        self.label_5.setGeometry(QtCore.QRect(110, 60, 401, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.label_5.setFont(font)
        self.label_5.setAutoFillBackground(False)
        self.label_5.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_5.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_5.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_5.setTextFormat(QtCore.Qt.AutoText)
        self.label_5.setWordWrap(True)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.page_3)
        self.label_6.setGeometry(QtCore.QRect(110, 150, 421, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_6.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_6.setTextFormat(QtCore.Qt.AutoText)
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.stackedWidget.addWidget(self.page_3)
        self.Page1 = QtWidgets.QPushButton(self.centralwidget)
        self.Page1.setGeometry(QtCore.QRect(280, 380, 75, 23))
        self.Page1.setStyleSheet("background-color: rgb(170, 170, 127);")
        self.Page1.setObjectName("Page1")
        self.Page2 = QtWidgets.QPushButton(self.centralwidget)
        self.Page2.setGeometry(QtCore.QRect(380, 380, 75, 23))
        self.Page2.setStyleSheet("background-color: rgb(0, 255, 127);")
        self.Page2.setObjectName("Page2")
        self.Page3 = QtWidgets.QPushButton(self.centralwidget)
        self.Page3.setGeometry(QtCore.QRect(480, 380, 75, 23))
        self.Page3.setStyleSheet("background-color: rgb(255, 170, 255);")
        self.Page3.setObjectName("Page3")
        self.start = QtWidgets.QPushButton(self.centralwidget)
        self.start.setGeometry(QtCore.QRect(180, 380, 75, 23))
        self.start.setStyleSheet("background-color: rgb(255, 170, 255);")
        self.start.setObjectName("start")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(80, 380, 75, 23))
        self.pushButton.setStyleSheet("background-color: rgb(255, 85, 255);")
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.label_5.setText(_translate("MainWindow", "Xin chào,bạn tên là gì"))
        self.label_6.setText(_translate("MainWindow", "Chào tôi là AiMon"))
        self.Page1.setText(_translate("MainWindow", "Page1"))
        self.Page2.setText(_translate("MainWindow", "Page2"))
        self.Page3.setText(_translate("MainWindow", "Page3"))
        self.start.setText(_translate("MainWindow", "Start"))
        self.pushButton.setText(_translate("MainWindow", "Exit"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())