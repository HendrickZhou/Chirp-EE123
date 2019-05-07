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
    def __init__(self, dataStream, dataInfo):
        """
        Take the image pixel data nparry as input.
        """
        self.dataStream = dataStream
        self.dataInfo = dataInfo

    #-------------------------------------#
    # FILE CODING
    #-------------------------------------#
    def codingData(self):
        self.bodyData
        self.headerData


    #-------------------------------------#
    # RESAMPLE
    #-------------------------------------#
    def encode(self, info, bodyData):
        """
        to save the trouble from python bitstream, we'll use file as the buffer for transmission
        """
        filename = asset_path + 'buffer.txt'
        with open(filename, 'bw+') as f_buffer:
            # encode the origin_info
            new_info = [len(info)] + info
            header = pack('%si' % len(new_info), *new_info)
            # flatten the numpy array and encode 
            dataVec = bodyData.tolist()
            body_header = pack('i', len(dataVec))
            # Judge if the len need to use long
            body = body_header + pack('%si' % len(dataVec), *dataVec)  
            f_buffer.write(header+body)

    def decode(self):
        filename = asset_path + 'buffer.txt'
        with open(filename, 'rb') as f_buffer:
            data = f_buffer.read()
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


    def upsample(npArray, info):
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

encode(info, r.flatten())
inf, bodyDat = decode()

com_height = inf[5]
com_width = inf[6]
com_channels = inf[7]
com_frames = inf[8]
recons = upsample(bodyDat.reshape(com_frames, com_height, com_width, com_channels), inf)
npArray_play(recons, frame_rate = 20)

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
