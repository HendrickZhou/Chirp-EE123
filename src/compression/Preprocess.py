import sys
import os
script_path = os.path.dirname(os.path.abspath( __file__ ))
module_path = script_path[:script_path.rfind('src')]+ 'src' + '/'
asset_path = script_path[:script_path.rfind('src')]+ 'asset' + '/'
sys.path.append(module_path)
from utils.Helper_functions import *
from compressData import CompressData
from loadData import LoadData

# print(script_path)

class Preprocess:
    """
    This class is the interface for everything

    Transmit:
        load .png -> compress and save it to buffer.jpg/txt -> load buffer file and transmit
    Receive:
        receive and save it to buffer file -> read buffer file and decomress it -> evaluate -> save .png
    """
    def __init__(self, filename):
        self.filename = filename
        self.loadData = LoadData()
        self.compressData = CompressData()


    #-------------------------------------#
    # TRANSMIT END
    #-------------------------------------#
    def processDataTran(self, method = 'downsample', params={'factor': 0.5}):

        self.loadData.loadVideoPixelData(self.filename)

        if method == 'downsample':
            r, info = self.compressData.downsample(self.loadData.image_stack, params["factor"])
            self.compressData.encode_resample(info, r.flatten())
        elif method == 'pca':
            pass

        # write the prefix and combine all handled data together and wirte to the file now
        self.compressData.encode(method)
        self.loadData.openFile(self.compressData.compressedFileName, 'rb')

    def readStream(self, chunkSize):
        bits = self.loadData.loadBitStream(chunkSize)
        return bits

    def closeStream(self):
        self.loadData.closeFile()

    #-------------------------------------#
    # RECEIVE END
    #-------------------------------------#
    def processDataRec(self):
        # two levels decode
        # 1. decide which compression method 
        # every method should put result to pixData numpy array
        self.compressData.decode(self.filename)
        method = self.compressData.method
        if method == 'downsample':
            # 2. in-method decode
            inf, bodyDat = self.compressData.decode_resample()
            com_height = inf[5]
            com_width = inf[6]
            com_channels = inf[7]
            com_frames = inf[8]
            recons = self.compressData.upsample(bodyDat.reshape(com_frames, com_height, com_width, com_channels), inf)
            npArray_play(recons, frame_rate = 20)
            pixData = recons
        elif method == 'pca':
            pass


        # save np array to png image file
        pngImg = Image.fromarray(pixData)
        pngImg.save(self.filename)


    #-------------------------------------#
    # EVALUATION
    #-------------------------------------# 
    def evaluate(self):
        pass





if __name__ == "__main__":

    # transmit end
    filename = 'simpson.png'
    method = 'downsample'
    params = {'factor':0.5}

    pro = Preprocess(asset_path+filename,)
    pro.processDataTran(method, params)
    for i in range(10):
        bitStream = pro.readStream(100)
        # transmit
    pro.closeStream()


    # receiver end
    # I can assum the transmission part has handled the filename already
    # it should be the same with original one or with extra part
    proR = Preprocess(asset_path+filename)

        
        
    filename = 'simpson_r.png' # for downsample use .txt, for other method use something like .jpg
    # what about if we don't know the compression method?
    # Judge from the header info
    proR.processDataRec()