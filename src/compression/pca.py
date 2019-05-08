import numpy as np
import matplotlib.pyplot as plt
from struct import pack,unpack
class PCA:
    def __init__(self,frames,framerate=10,compressionRatio = 0.005,sectionSize=80,minVar = 0.009):
        assert len(frames.shape)==4 #frames should be 4-D array
        self.frames=frames
        self.nImages, self.nRow, self.nCol, self.nColors = frames.shape  # assumes 3 color channel when preprocess

        # self.param={}
        # self.param['nImages'] = self.nImages
        # self.param['nRow'] = self.nRow
        # self.param['nCol'] = self.nCol
        # self.param['nColors'] = self.nColors
        self.param=[]
        self.param.append(self.nImages)
        self.param.append(self.nRow)
        self.param.append(self.nCol)
        self.param.append(self.nColors)

        self.compressionRation=compressionRatio
        self.sectionSize = min(sectionSize, int(self.nRow / 2), int(self.nCol / 2))
        self.minVar=minVar

        self.nRowSec = int(self.nRow / sectionSize)
        self.rowSec = np.arange(self.nRowSec + 1) * sectionSize
        self.rowSec[-1] = max(self.rowSec[-1], self.nRow)

        self.nColSec = int(self.nCol / sectionSize)
        self.colSec = np.arange(self.nColSec + 1) * self.sectionSize
        self.colSec[-1] = max(self.colSec[-1], self.nCol)

        # self.param['nRowSec'] = self.nRowSec
        # self.param['rowSec'] = self.rowSec
        # self.param['nColSec'] = self.nColSec
        # self.param['colSec'] = self.colSec
        self.param.append((len(self.rowSec)))
        self.param.append(self.nRowSec)
        self.param.extend(self.rowSec)
        self.param.append(framerate)
        self.param.append(len(self.colSec))
        self.param.append(self.nColSec)
        self.param.extend(self.colSec)

        self.nSec = self.nColSec * self.nRowSec
        self.segmentedX=None
        self.pca_result=None

    def procInput_noFlatten(self):
        segmentedX = [None] * (self.nSec)
        for ri in range(self.nRowSec):
            for ci in range(self.nColSec):
                rowSecStart = self.rowSec[ri]
                rowSecEnd = self.rowSec[ri + 1]
                colSecStart = self.colSec[ci]
                colSecEnd = self.colSec[ci + 1]

                nrow = (rowSecEnd - rowSecStart) * self.nColors
                ncol = (colSecEnd - colSecStart)
                seci = np.zeros((self.nImages * nrow, ncol))
                for img in range(self.nImages):
                    r = self.frames[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 0]
                    g = self.frames[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 1]
                    b = self.frames[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 2]

                    seci[img * nrow:(img + 1) * nrow, :] = np.concatenate((r, g, b), axis=0)
                # print(seci.shape)
                segmentedX[ri * self.nColSec + ci] = seci
                # print(ri*nColSec+ci)
        self.segmentedX=segmentedX

    def pca(self,x):
        """
        :param x: input n*d
        :return:(V,Y,Mx,w); V - eigenvector matrix (eigenvectors in columns); Y - coordinates after PCA; Mx - row mean of X, w - vector of eigenvalues
        """
        Mx = np.mean(x, axis=1)
        x_centered = np.copy(x)
        x_centered = x_centered - np.tile(np.reshape(Mx, (Mx.size, 1)), x.shape[1])

        cov = np.transpose(x_centered) @ x_centered
        # print(cov.shape)

        w, V = np.linalg.eig(cov)
        orderW = np.argsort(w * (-1))  # idx that sorts w in decending order
        w = w[orderW]
        V = V[:, orderW]

        # Vt=np.transpose(V)
        Y = x_centered @ V
        self.pca_result=(V, Y, Mx, w)
        return (V, Y, Mx, w)

    def pca_compression(self,x):
        V, Y, Mx, w = self.pca(x)
        w_varCap = w / np.sum(w)
        if self.minVar != None:
            idx = np.greater(w_varCap, self.minVar)
            V = V[:, idx]
            Y = Y[:, idx]
        # if self.compressionRation != None:
        #     V = V[:, :nPC]
        #     Y = Y[:, :nPC]
        return (V, Y, Mx, w)

    def getArraysToTransmit(self):
        compressedX = None

        # self.param['lenSegmentedX'] = len(self.segmentedX)
        self.param.append(len(self.segmentedX))
        print(len(self.segmentedX))
        for s in range(len(self.segmentedX)):
            # V,Y,Mx,w=pca_compression(segmentedX[s],int(segmentedX[s].shape[0]*compressionRatio))
            V, Y, Mx, w = self.pca_compression(self.segmentedX[s])

            # self.param['shapeV' + str(s)] = V.shape
            # self.param['shapeY' + str(s)] = Y.shape
            # self.param['shapeMx' + str(s)] = Mx.shape
            self.param.append(V.shape[0])
            self.param.append(V.shape[1])
            self.param.append(Y.shape[0])
            self.param.append(Y.shape[1])
            self.param.append(Mx.shape[0])

            V=V*1e4
            V=V.astype('int')
            Y=Y.astype('int')
            Mx=Mx.astype('int')

            ci = np.concatenate((V.flatten(), Y.flatten(), Mx.flatten()))
            if s == 0:
                compressedX = ci
            else:
                compressedX = np.concatenate((compressedX, ci))
        # print(np.size(compressedX))
        return compressedX.tolist()

    def encode_PCA(self,compressedX):
        """
        to save the trouble from python bitstream, we'll use file as the buffer for transmission
        """

        lstTotransmit=[len(self.param)]
        lstTotransmit.extend(self.param)
        lstTotransmit.append(len(compressedX))
        lstTotransmit.extend(compressedX)

        return pack('%si'%len(lstTotransmit),*lstTotransmit)

def decode_PCA(binaryArr):
    # extract param
    nParam=unpack('%si'%1,binaryArr[:4])[0]
    binaryArr=binaryArr[4:]
    paramslen=4*nParam
    params=unpack('%si'%nParam,binaryArr[:paramslen])
    binaryArr=binaryArr[paramslen:]

    param_dict={}
    param_dict['nImages']=params[0]
    param_dict['nRow']=params[1]
    param_dict['nCol']=params[2]
    param_dict['nColors']=params[3]

    nRowSec=params[4]
    param_dict['nRowSec']=params[5]
    param_dict['rowSec']=params[6:6+nRowSec]
    # print(params[5:5+nRowSec])
    idx=6+nRowSec
    param_dict['frameRate']=params[idx]
    idx+=1
    nColSec=params[idx]
    idx+=1
    param_dict['nColSec']=params[idx]
    idx+=1
    param_dict['colSec']=params[idx:idx+nColSec]
    # print(nColSec)
    idx=idx+nColSec
    lenSegmentedX=params[idx]
    param_dict['lenSegmentedX']=lenSegmentedX
    idx+=1
    print(lenSegmentedX)
    for s in range(lenSegmentedX):
        param_dict['shapeV' + str(s)]=(params[idx],params[idx+1])
        idx+=2
        param_dict['shapeY' + str(s)]=(params[idx],params[idx+1])
        idx+=2
        param_dict['shapeMx' + str(s)]=(params[idx])
        idx+=1

    # extract compressedX
    nCompressedX=unpack('%si'%1,binaryArr[:4])[0]
    # print(nCompressedX)
    binaryArr=binaryArr[4:]
    nElements=4*nCompressedX
    compressedX=unpack('%si'%nCompressedX,binaryArr[:nElements])


    return list(compressedX), param_dict


def pca_reconstruction(V, Y, Mx):
    V=V/1e4
    x_centered = Y @ np.transpose(V)
    x = x_centered + np.tile(np.reshape(Mx, (Mx.size, 1)), x_centered.shape[1])
    return x

def reassembleX(x_segmented,nImages,nRow,nCol,nColors,nRowSec,nColSec,rowSec,colSec):
    x=np.zeros((nImages,nRow,nCol,nColors))
    for ri in range(nRowSec):
        for ci in range(nColSec):
            rowSecStart = rowSec[ri]
            rowSecEnd = rowSec[ri + 1]
            colSecStart = colSec[ci]
            colSecEnd = colSec[ci + 1]

            # print(colSecStart)
            # print(colSecEnd)

            nrow = (rowSecEnd - rowSecStart) * nColors
            nrowPerColor=(rowSecEnd - rowSecStart)
            ncol = (colSecEnd - colSecStart)
            seci = x_segmented[ri * nColSec + ci]
            for img in range(nImages):
                print(img)
                img_seci=seci[img*nrow:(img+1)*nrow, :]
                # print(img_seci.shape)
                x[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 0]=img_seci[:nrowPerColor,:]
                x[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 1]=img_seci[nrowPerColor:nrowPerColor*2,:]
                x[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 2]=img_seci[nrowPerColor*2:nrowPerColor*3,:]

    return x

def pca_reconstruct(compressedX,param):
    lenSegmentedX=param['lenSegmentedX']
    nImages=param['nImages']
    nRow=param['nRow']
    nCol=param['nCol']
    nColors=param['nColors']
    nRowSec=param['nRowSec']
    rowSec=param['rowSec']
    nColSec=param['nColSec']
    colSec=param['colSec']


    reconstructX=[None]*lenSegmentedX
    for s in range(lenSegmentedX):
        print(s)
        Vs_shape=param['shapeV' + str(s)]
        Vs_size=Vs_shape[0]*Vs_shape[1]
        Ys_shape=param['shapeY' + str(s)]
        Ys_size=Ys_shape[0]*Ys_shape[1]
        Mx_shape=param['shapeMx' + str(s)]
        Mx_size=Mx_shape

        V=np.reshape(compressedX[:Vs_size],Vs_shape)
        compressedX=compressedX[Vs_size:]

        Y=np.reshape(compressedX[:Ys_size],Ys_shape)
        compressedX=compressedX[Ys_size:]

        Mx=np.reshape(compressedX[:Mx_size],Mx_shape)
        compressedX=compressedX[Mx_size:]


        reconstructedXs=pca_reconstruction(V,Y,Mx)
        reconstructX[s]=reconstructedXs
        # print('mse per pixel:'+str(metrics.mean_squared_error(segmentedX[s],reconstructedX)/nPixels))
    print('done')
    reassembledImg=reassembleX(reconstructX,nImages,nRow,nCol,nColors,nRowSec,nColSec,rowSec,colSec)

    for img in range(nImages):
        reassembledImg[img]=reassembledImg[img]/np.max(reassembledImg[img])
        plt.imshow(reassembledImg[img])
        # plt.imshow(frames.image_stack[img])
        plt.savefig("frame"+str(img)+'.png')
        plt.show()


    return reassembledImg