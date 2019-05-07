import sys
import os
script_path = os.path.dirname(os.path.abspath( __file__ ))
module_path = script_path[:script_path.rfind('src')]+ 'src' + '/'
asset_path = script_path[:script_path.rfind('src')]+ 'asset' + '/'
sys.path.append(module_path)
from scipy import signal 
from scipy import ndimage, misc, interpolate
from struct import *
from utils.Helper_functions import *
from loadData import LoadData

class CompressData:
    """
    This class contains methods for image/video compression

    attributes:
        bodyData
        headerData
        finalData




    """
    def __init__(self):
        """
        Take the image pixel data nparry as input.
        """
        # self.dataStream = dataStream
        # self.dataInfo = dataInfo
        image = np.array(Image.open(buffer_path))
        pass

    #-------------------------------------#
    # FILE CODING
    # this part only handle with already encoded file and add prefix and write it to file
    #-------------------------------------#
    def encode(self, method):
        # add prefix to our 
        mainData = self.encode_resample()
        pass
    def decode(self, filename):
        self.compressedFileName = filename
        # open and read file
        with open(filename, 'rb') as f_buffer:
            data = f_buffer.read()
        # extract the method

        # get remaining main data for further process
        self.method = 'resample'
        self.mainData = mainData
        # return method, mainData

    #-------------------------------------#
    # RESAMPLE
    #-------------------------------------#
    def encode_resample(self, info, bodyData):
        """
        to save the trouble from python bitstream, we'll use file as the buffer for transmission
        """
        filename = asset_path + 'buffer.txt'
        self.compressedFileName = filename
        # with open(filename, 'bw+') as self.f_buffer:
            # encode the origin_info


            # add the file suffix!!!!!



        new_info = [len(info)] + info
        header = pack('%si' % len(new_info), *new_info)
        # flatten the numpy array and encode 
        dataVec = bodyData.tolist()
        body_header = pack('i', len(dataVec))
        # Judge if the len need to use long
        body = body_header + pack('%si' % len(dataVec), *dataVec)  
            # self.f_buffer.write(header+body)

        return header + body

    def decode_resample(self):
        # filename = asset_path + 'buffer.txt'
        # self.compressedFileName = filename
        data = self.mainData
        # add the file suffix!!!! and filename

        # decode the origin_info
        header_len = unpack('i', data[0:4])
        header_end_idx = 4*header_len[0]+4
        info = unpack('%si' % (header_len[0]), data[4: header_end_idx])
        # decode body
        body_start_idx = header_end_idx
        body_len = unpack('i', data[body_start_idx: body_start_idx + 4])
        bodyData = np.array(unpack('%si' % (body_len[0]), data[body_start_idx + 4:]))
        return info, bodyData
        
    
    def downsample(self, npArray, factor_xy, timeFlag = False):
        """
        Params:
            timeFlag: whether or not downsample in t index, False by default
            factor_xy: only support float, if wrong type, use 100% by default. 
        Notice:
            If you want to use 100%, use 1.0 instead of 1!!
            we only support resample by 2 on time axis!!
        """
        if type(factor_xy) is not float:
            print("wrong sampling rate format!!!, continue with factor_xy = 1")
            factor_xy = 1.0
        
        if len(npArray.shape) == 3:
            ori_height= npArray.shape[0]
            ori_width = npArray.shape[1]
            ori_channels = npArray.shape[2]
            timeFlag = False
        else:
            ori_frames = npArray.shape[0]
            ori_height= npArray.shape[1]
            ori_width = npArray.shape[2]
            ori_channels = npArray.shape[3]
            
        height = int(ori_height*factor_xy)
        width = int(ori_width*factor_xy)
            
        data_xy = np.empty([ori_frames, height, width, ori_channels], dtype = 'uint8')
        for i in range(ori_frames):
            data_xy[i] = misc.imresize(npArray[i], factor_xy)
        
        # time axis
        if timeFlag:
            # downsample
            h_t = signal.firwin(ori_frames, 1/2)
            ndimage.convolve1d(data_xy, h_t, axis = 0)

            new_frames = int(np.ceil(ori_frames/2))
            data_t = np.empty([new_frames, height, width, ori_channels], dtype = 'uint8')

            for i in range(ori_frames):
                if i%2 == 0: 
                    data_t[i//2] = data_xy[i]
            result = data_t
            frames = new_frames
        else:
            result = data_xy
            frames = ori_frames

        origin_info = [ori_height, ori_width, ori_frames, ori_channels]
        compressed_info = [timeFlag, height, width, frames, ori_channels]
        info = origin_info+compressed_info
        
        return result, info


    def upsample(self, npArray, info):
        """    
        origin_info: list
        """
        frames = npArray.shape[0]
        height= npArray.shape[1]
        width = npArray.shape[2]
        channels = npArray.shape[3]
            
        ori_height = info[0]
        ori_width = info[1]
        ori_frames = info[2]
        ori_channels = info[3]
        timeFlag = info[4]
            
        data_t = np.empty([ori_frames, height, width, ori_channels], dtype = 'uint8')
        if timeFlag:
                # upsample
                for i in range(ori_frames):
                    if i%2 == 0:
                        data_t[i] = npArray[i//2]
                    else:
                        data_t[i] = np.zeros([height, width, channels], dtype = 'uint8')
                data_t = signal.resample(data_t, ori_frames, axis = 0, )                
        else:
            data_t = npArray

        result = np.empty([ori_frames, ori_height, ori_width, ori_channels], dtype = 'uint8')
        for i in range(ori_frames):
            result[i] = misc.imresize(data_t[i], [ori_height, ori_width])
        
        return result

    #-------------------------------------#
    # PCA
    #-------------------------------------#
    def PCA(self):
        pass


    #-------------------------------------#
    # JPEG
    #-------------------------------------#
    def JPEG(self):
        pass
    
    #-------------------------------------#
    # JPEG 2000
    #-------------------------------------#
    def JPEG2000(self):
        pass

    #-------------------------------------#
    # MPG
    #-------------------------------------#
    def MPG(self):
        pass

    #-------------------------------------#
    # Motion Vector
    #-------------------------------------#
    def MoVec(self):
        pass






if __name__ == "__main__":
    loadData = LoadData()
