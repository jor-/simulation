# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created: Sun Jun 30 21:34:46 2013
#      by: PyQt4 UI code generator 4.10
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(423, 419)
        self.pushButton = QtGui.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(160, 380, 81, 28))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.formLayoutWidget = QtGui.QWidget(Dialog)
        self.formLayoutWidget.setGeometry(QtCore.QRect(20, 20, 381, 181))
        self.formLayoutWidget.setObjectName(_fromUtf8("formLayoutWidget"))
        self.formLayout = QtGui.QFormLayout(self.formLayoutWidget)
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setMargin(0)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label_2 = QtGui.QLabel(self.formLayoutWidget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_2)
        self.comboBox_2 = QtGui.QComboBox(self.formLayoutWidget)
        self.comboBox_2.setObjectName(_fromUtf8("comboBox_2"))
        self.comboBox_2.addItem(_fromUtf8(""))
        self.comboBox_2.addItem(_fromUtf8(""))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.comboBox_2)
        self.label_3 = QtGui.QLabel(self.formLayoutWidget)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.label_3)
        self.comboBox_3 = QtGui.QComboBox(self.formLayoutWidget)
        self.comboBox_3.setObjectName(_fromUtf8("comboBox_3"))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.comboBox_3)
        self.label_4 = QtGui.QLabel(self.formLayoutWidget)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.label_4)
        self.comboBox_4 = QtGui.QComboBox(self.formLayoutWidget)
        self.comboBox_4.setObjectName(_fromUtf8("comboBox_4"))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.comboBox_4.addItem(_fromUtf8(""))
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.comboBox_4)
        self.label_5 = QtGui.QLabel(self.formLayoutWidget)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.formLayout.setWidget(5, QtGui.QFormLayout.LabelRole, self.label_5)
        self.comboBox_5 = QtGui.QComboBox(self.formLayoutWidget)
        self.comboBox_5.setObjectName(_fromUtf8("comboBox_5"))
        self.comboBox_5.addItem(_fromUtf8(""))
        self.comboBox_5.addItem(_fromUtf8(""))
        self.comboBox_5.addItem(_fromUtf8(""))
        self.comboBox_5.addItem(_fromUtf8(""))
        self.comboBox_5.addItem(_fromUtf8(""))
        self.comboBox_5.addItem(_fromUtf8(""))
        self.comboBox_5.addItem(_fromUtf8(""))
        self.comboBox_5.addItem(_fromUtf8(""))
        self.comboBox_5.addItem(_fromUtf8(""))
        self.comboBox_5.addItem(_fromUtf8(""))
        self.comboBox_5.addItem(_fromUtf8(""))
        self.comboBox_5.addItem(_fromUtf8(""))
        self.comboBox_5.addItem(_fromUtf8(""))
        self.comboBox_5.addItem(_fromUtf8(""))
        self.formLayout.setWidget(5, QtGui.QFormLayout.FieldRole, self.comboBox_5)
        self.label = QtGui.QLabel(self.formLayoutWidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout.setWidget(6, QtGui.QFormLayout.LabelRole, self.label)
        self.comboBox = QtGui.QComboBox(self.formLayoutWidget)
        self.comboBox.setObjectName(_fromUtf8("comboBox"))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.formLayout.setWidget(6, QtGui.QFormLayout.FieldRole, self.comboBox)
        self.formLayoutWidget_2 = QtGui.QWidget(Dialog)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(20, 210, 381, 161))
        self.formLayoutWidget_2.setObjectName(_fromUtf8("formLayoutWidget_2"))
        self.formLayout_2 = QtGui.QFormLayout(self.formLayoutWidget_2)
        self.formLayout_2.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout_2.setMargin(0)
        self.formLayout_2.setObjectName(_fromUtf8("formLayout_2"))
        self.label_8 = QtGui.QLabel(self.formLayoutWidget_2)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_8)
        self.label_7 = QtGui.QLabel(self.formLayoutWidget_2)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.formLayout_2.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_7)
        self.parameter_label_2 = QtGui.QLabel(self.formLayoutWidget_2)
        self.parameter_label_2.setObjectName(_fromUtf8("parameter_label_2"))
        self.formLayout_2.setWidget(1, QtGui.QFormLayout.FieldRole, self.parameter_label_2)
        self.label_6 = QtGui.QLabel(self.formLayoutWidget_2)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.formLayout_2.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_6)
        self.parameter_label_3 = QtGui.QLabel(self.formLayoutWidget_2)
        self.parameter_label_3.setObjectName(_fromUtf8("parameter_label_3"))
        self.formLayout_2.setWidget(2, QtGui.QFormLayout.FieldRole, self.parameter_label_3)
        self.label_10 = QtGui.QLabel(self.formLayoutWidget_2)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.formLayout_2.setWidget(3, QtGui.QFormLayout.LabelRole, self.label_10)
        self.parameter_label_4 = QtGui.QLabel(self.formLayoutWidget_2)
        self.parameter_label_4.setObjectName(_fromUtf8("parameter_label_4"))
        self.formLayout_2.setWidget(3, QtGui.QFormLayout.FieldRole, self.parameter_label_4)
        self.label_9 = QtGui.QLabel(self.formLayoutWidget_2)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.formLayout_2.setWidget(4, QtGui.QFormLayout.LabelRole, self.label_9)
        self.parameter_label_5 = QtGui.QLabel(self.formLayoutWidget_2)
        self.parameter_label_5.setObjectName(_fromUtf8("parameter_label_5"))
        self.formLayout_2.setWidget(4, QtGui.QFormLayout.FieldRole, self.parameter_label_5)
        self.label_12 = QtGui.QLabel(self.formLayoutWidget_2)
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.formLayout_2.setWidget(5, QtGui.QFormLayout.LabelRole, self.label_12)
        self.parameter_label_6 = QtGui.QLabel(self.formLayoutWidget_2)
        self.parameter_label_6.setObjectName(_fromUtf8("parameter_label_6"))
        self.formLayout_2.setWidget(5, QtGui.QFormLayout.FieldRole, self.parameter_label_6)
        self.label_11 = QtGui.QLabel(self.formLayoutWidget_2)
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.formLayout_2.setWidget(6, QtGui.QFormLayout.LabelRole, self.label_11)
        self.parameter_label_7 = QtGui.QLabel(self.formLayoutWidget_2)
        self.parameter_label_7.setObjectName(_fromUtf8("parameter_label_7"))
        self.formLayout_2.setWidget(6, QtGui.QFormLayout.FieldRole, self.parameter_label_7)
        self.parameter_label_1 = QtGui.QLabel(self.formLayoutWidget_2)
        self.parameter_label_1.setObjectName(_fromUtf8("parameter_label_1"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.FieldRole, self.parameter_label_1)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.pushButton, QtCore.SIGNAL(_fromUtf8("clicked()")), Dialog.draw_plot)
        QtCore.QObject.connect(self.comboBox, QtCore.SIGNAL(_fromUtf8("currentIndexChanged(int)")), Dialog.set_parameter_set)
        QtCore.QObject.connect(self.comboBox_2, QtCore.SIGNAL(_fromUtf8("currentIndexChanged(int)")), Dialog.set_tracer_index)
        QtCore.QObject.connect(self.comboBox_3, QtCore.SIGNAL(_fromUtf8("currentIndexChanged(int)")), Dialog.set_time_index)
        QtCore.QObject.connect(self.comboBox_4, QtCore.SIGNAL(_fromUtf8("currentIndexChanged(int)")), Dialog.set_depth_index)
        QtCore.QObject.connect(self.comboBox_5, QtCore.SIGNAL(_fromUtf8("currentIndexChanged(int)")), Dialog.set_plot_index)
        QtCore.QObject.connect(Dialog, QtCore.SIGNAL(_fromUtf8("parameter_str_changed(QString)")), self.parameter_label_1.setText)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.pushButton.setText(_translate("Dialog", "draw plots", None))
        self.label_2.setText(_translate("Dialog", "tracer: ", None))
        self.comboBox_2.setItemText(0, _translate("Dialog", "DOP", None))
        self.comboBox_2.setItemText(1, _translate("Dialog", "PO4", None))
        self.label_3.setText(_translate("Dialog", "time: ", None))
        self.comboBox_3.setItemText(0, _translate("Dialog", "annual (averaged)", None))
        self.comboBox_3.setItemText(1, _translate("Dialog", "January", None))
        self.comboBox_3.setItemText(2, _translate("Dialog", "February", None))
        self.comboBox_3.setItemText(3, _translate("Dialog", "March", None))
        self.comboBox_3.setItemText(4, _translate("Dialog", "April", None))
        self.comboBox_3.setItemText(5, _translate("Dialog", "May", None))
        self.comboBox_3.setItemText(6, _translate("Dialog", "June", None))
        self.comboBox_3.setItemText(7, _translate("Dialog", "July", None))
        self.comboBox_3.setItemText(8, _translate("Dialog", "August", None))
        self.comboBox_3.setItemText(9, _translate("Dialog", "September", None))
        self.comboBox_3.setItemText(10, _translate("Dialog", "October", None))
        self.comboBox_3.setItemText(11, _translate("Dialog", "November", None))
        self.comboBox_3.setItemText(12, _translate("Dialog", "December", None))
        self.label_4.setText(_translate("Dialog", "depth: ", None))
        self.comboBox_4.setItemText(0, _translate("Dialog", "0m - 50m", None))
        self.comboBox_4.setItemText(1, _translate("Dialog", "50m - 120m", None))
        self.comboBox_4.setItemText(2, _translate("Dialog", "120m - 220m", None))
        self.comboBox_4.setItemText(3, _translate("Dialog", "220m - 360m", None))
        self.comboBox_4.setItemText(4, _translate("Dialog", "360m - 550m", None))
        self.comboBox_4.setItemText(5, _translate("Dialog", "550m - 790m", None))
        self.comboBox_4.setItemText(6, _translate("Dialog", "790m - 1080m", None))
        self.comboBox_4.setItemText(7, _translate("Dialog", "1080m - 1420m", None))
        self.comboBox_4.setItemText(8, _translate("Dialog", "1420m - 1810m", None))
        self.comboBox_4.setItemText(9, _translate("Dialog", "1810m - 2250m", None))
        self.comboBox_4.setItemText(10, _translate("Dialog", "2250m - 2740m", None))
        self.comboBox_4.setItemText(11, _translate("Dialog", "2740m - 3280m", None))
        self.comboBox_4.setItemText(12, _translate("Dialog", "3280m - 3870m", None))
        self.comboBox_4.setItemText(13, _translate("Dialog", "3870m - 4510m", None))
        self.comboBox_4.setItemText(14, _translate("Dialog", "below 4510m", None))
        self.label_5.setText(_translate("Dialog", "plot: ", None))
        self.comboBox_5.setItemText(0, _translate("Dialog", "model: output", None))
        self.comboBox_5.setItemText(1, _translate("Dialog", "model: accuracy", None))
        self.comboBox_5.setItemText(2, _translate("Dialog", "model: difference output and observations", None))
        self.comboBox_5.setItemText(3, _translate("Dialog", "observations: mean", None))
        self.comboBox_5.setItemText(4, _translate("Dialog", "observations: number", None))
        self.comboBox_5.setItemText(5, _translate("Dialog", "observations: variance of measurement errors", None))
        self.comboBox_5.setItemText(6, _translate("Dialog", "sensitivity: all parameters", None))
        self.comboBox_5.setItemText(7, _translate("Dialog", "sensitivity: Lambda (remineralization rate DOP)", None))
        self.comboBox_5.setItemText(8, _translate("Dialog", "sensitivity: Alpha (max production rate)", None))
        self.comboBox_5.setItemText(9, _translate("Dialog", "sensitivity: Sigma (fraction of DOP)", None))
        self.comboBox_5.setItemText(10, _translate("Dialog", "sensitivity: K_N (half saturation N)", None))
        self.comboBox_5.setItemText(11, _translate("Dialog", "sensitivity: K_I (half saturation light)", None))
        self.comboBox_5.setItemText(12, _translate("Dialog", "sensitivity: K_{H_2O} (attenuation water)", None))
        self.comboBox_5.setItemText(13, _translate("Dialog", "sensitivity: b (sinking exponent)", None))
        self.label.setText(_translate("Dialog", "parameter set: ", None))
        self.comboBox.setItemText(0, _translate("Dialog", "0", None))
        self.comboBox.setItemText(1, _translate("Dialog", "1", None))
        self.comboBox.setItemText(2, _translate("Dialog", "2", None))
        self.comboBox.setItemText(3, _translate("Dialog", "3", None))
        self.label_8.setText(_translate("Dialog", "Lambda: remineralization rate DOP = ", None))
        self.label_7.setText(_translate("Dialog", "Alpha: max production rate = ", None))
        self.parameter_label_2.setText(_translate("Dialog", "TextLabel", None))
        self.label_6.setText(_translate("Dialog", "Sigma: fraction of DOP = ", None))
        self.parameter_label_3.setText(_translate("Dialog", "TextLabel", None))
        self.label_10.setText(_translate("Dialog", "K_N: half saturation N = ", None))
        self.parameter_label_4.setText(_translate("Dialog", "TextLabel", None))
        self.label_9.setText(_translate("Dialog", "K_I: half saturation light = ", None))
        self.parameter_label_5.setText(_translate("Dialog", "TextLabel", None))
        self.label_12.setText(_translate("Dialog", "K_{H_2O}: attenuation water = ", None))
        self.parameter_label_6.setText(_translate("Dialog", "TextLabel", None))
        self.label_11.setText(_translate("Dialog", "b: sinking exponent = ", None))
        self.parameter_label_7.setText(_translate("Dialog", "TextLabel", None))
        self.parameter_label_1.setText(_translate("Dialog", "TextLabel", None))

