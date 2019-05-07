import sys
import os
script_path = os.path.dirname(os.path.abspath( __file__ ))
module_path = script_path[:script_path.rfind('src')]+ 'src' + '/'
sys.path.append(module_path)
from utils.Helper_functions import *

class LoadData:
    """
    This class loads all kinds of image/video file types

    First of all we only need to handle with the 3 channels PNG  

    load the data as a file for transmission
    load the data, get the pixel matrix and perform the compression stuff

    We can load 4 channel image though. But we don't need to handle it for simplicity
    """

    def __init__(self):
        pass    

    def loadVideoPixelData(self, filename):
        """
        return the 3d nparray of image matrix stack
        absolute path and relative path is both supported

        * right now only support absolute path
        """

        self.image_stack = imageStack_load(filename) # uint8 np array

    # def saveVideoPixelData(self, filename):
    #     pass

    # def extractInfo(self):
    #     pass


    #-------------------------------------#
    # debug pipelines
    #-------------------------------------#
    def loadImagePixelData(self, filename):
        """
        If you only want to handle with single image, and try to avoid creating extra image file
        """
        # if not single image
        self.image = np.array(Image.open(filename))
        return self.image

    def openFile(self, filename, mode = 'wb'):
        # try except
        self.file = open(filename, mode)
        self.streamFlag = True

    def closeFile(self):
        # if closed already, doesn't matter
        self.file.close()
        self.streamFlag = False


    def saveBitStream(self, chunkSize):
        

    def loadBitStream(self, chunkSize):
        """
        load the file stream without any converion

        to prevent running out of memory, send them by chunks
        """
        if self.streamFlag == False:
            print("The file stream is already empty, returning empty bitarray")
            self.closeFile()
            return b''

        # try exception
        # catch if no file is open correctly
        bitStream = self.file.read(chunkSize)
        if bitStream == b'':
            print("The file stream is empty now")
            self.closeFile()
        return bitStream


if __name__ == "__main__":
    loadData1 = LoadData()
    loadData2 = LoadData()

    loadData1.openFile("/Users/zhouhang/Project/Chirp-EE123/asset/simpson.png", 'rb')
    for i in range(1, 10):
        bits = loadData1.loadBitStream(100)
        print(bits)
    loadData1.closeFile()

    loadData1.openFile("/Users/zhouhang/Project/Chirp-EE123/asset/test.tiff", 'rb')
    for i in range(1, 10):
        bits = loadData1.loadBitStream(100)
        print(bits)
    loadData1.closeFile()

    loadData2.loadImagePixelData('/Users/zhouhang/Project/Chirp-EE123/asset/test.tiff')
    loadData2.loadVideoPixelData('/Users/zhouhang/Project/Chirp-EE123/asset/Andy_Video.png')
    
    print("shape of image data %s"%(loadData2.image.shape,))
    print("shape of video data %s"%(loadData2.image_stack.shape,))