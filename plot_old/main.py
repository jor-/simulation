import sys
import argparse

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from ndop.plot.logic import Logic
from ndop.plot.gui import Ui_Dialog


class MainWindow(QWidget, Logic, Ui_Dialog):

    def __init__(self, debug_level=0):
        Logic.__init__(self, debug_level)
        QWidget.__init__(self)
        # set up User Interface (widgets, layout...)
        self.setupUi(self)

    
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