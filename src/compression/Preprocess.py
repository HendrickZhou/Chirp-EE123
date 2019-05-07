import sys
import os
script_path = os.path.dirname(os.path.abspath( __file__ ))
module_path = script_path[:script_path.rfind('src')]+ 'src' + '/'
sys.path.append(module_path)
from utils.Helper_functions import *
# from compression import CompressData

# print(script_path)

class Preprocess:
    """
    This class is the interface for everything

    """
    def __init__(self):
        self.loadData = LoadData()
        self.compressData = CompressData()

