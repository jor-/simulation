import sys
import argparse

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from ndop.plot_gui.logic import Logic
from ndop.plot_gui.gui import Ui_Dialog


class MainWindow(QWidget, Logic, Ui_Dialog):

    def __init__(self, debug_level=0):
        QWidget.__init__(self)
        self.setupUi(self)
        
        Logic.__init__(self, debug_level)
        # set up User Interface (widgets, layout...)
        
    
    def set_parameter_set(self, parameter_set):
        super(MainWindow, self).set_parameter_set(parameter_set)
#         self.parameter_str_changed(str(parameter_set))
        parameters_strings = self.get_parameters_strings()
        self.parameter_label_1.setText(parameters_strings[0])
        self.parameter_label_2.setText(parameters_strings[1])
        self.parameter_label_3.setText(parameters_strings[2])
        self.parameter_label_4.setText(parameters_strings[3])
        self.parameter_label_5.setText(parameters_strings[4])
        self.parameter_label_6.setText(parameters_strings[5])
        self.parameter_label_7.setText(parameters_strings[6])

    
# Main entry to program.  Sets up the main app and create a new window.
def main(argv):
    # parse argv
    parser = argparse.ArgumentParser(description='GUI for plotting sensitivity of NDOP.')
    parser.add_argument('-d', '--debug_level', default=0, type=int, help='Increase the debug level for more debug informations.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    args = parser.parse_args()
    
    # create Qt application
    app = QApplication(argv,True)
 
    # create main window
    wnd = MainWindow(args.debug_level) # classname
    wnd.show()
    
    # Connect signal for app finish
    app.connect(app, SIGNAL("lastWindowClosed()"), app, SLOT("quit()"))
    
    # Start the app up
    sys.exit(app.exec_())
 
if __name__ == "__main__":
    main(sys.argv)