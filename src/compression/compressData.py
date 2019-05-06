import sys
import os
script_path = os.path.dirname(os.path.abspath( __file__ ))
module_path = script_path[:script_path.rfind('src')]+ 'src' + '/'
sys.path.append(module_path)
from utils.Helper_functions import *
from loadData import LoadData

class CompressData:
    """
    This class contains methods for image/video compression
    """
    def __init__(self, dataStream, dataInfo):
        """
        Take the image pixel data nparry as input.
        """
        self.dataStream = dataStream
        self.dataInfo = dataInfo

    def resample(self):

        pass

    def PCA(self):
        pass


    def JPEG(self):
        pass
    
    def JPEG2000(self):
        pass

    def MPG(self):
        pass

    def MoVec(self):
        pass





if __name__ == "__main__":
    loadData = LoadData()
