import sys
import os
script_path = os.path.abspath('')
module_path = script_path[:script_path.rfind('src')]+ 'src' + '/'
asset_path = script_path[:script_path.rfind('src')]+ 'asset' + '/'
sys.path.append(module_path)
import numpy as np
import matplotlib.pyplot as plt
from utils.Helper_functions import *
from scipy import signal



class PCA:

    def __init__(self):
        pass

    def init_rec(self, nImages,nRow,nCol,nColors):
        param={}
        param['nImages']=nImages
        param['nRow']=nRow
        param['nCol']=nCol
        param['nColors']=nColors

        self.nRow=nRow
        self.nCol=nCol
        self.nColors=nColors
        self.nImages = nImages

        self.param = param

        #tunable parameters
        self.compressionRatio=0.005
        self.sectionSize=min(80,int(nRow/2),int(nCol/2))
        self.minSectionSize=10
        self.minSection=2
        self.nPC=None
        self.minVar=0.009

        self.ifFlatten=False #if flatten: each colored section is a 1d vector input; otherwise: each monochrome row in a section is a separate 1d input

        self.calParam()
        self.calParam2()
        self.find_sections()
        self.getCompressedX()

    def init(self, image_stack):
        self.image_stack = image_stack

        # image_stack = imageStack_load(asset_path+'simpson.png')


        # print("shape of video data %s" % (image_stack.shape,))
        nImages,nRow,nCol,nColors=image_stack.shape      #assumes 3 color channel when preprocess
        param={}
        param['nImages']=nImages
        param['nRow']=nRow
        param['nCol']=nCol
        param['nColors']=nColors

        self.nRow=nRow
        self.nCol=nCol
        self.nColors=nColors
        self.nImages = nImages

        self.param = param

        #tunable parameters
        self.compressionRatio=0.005
        self.sectionSize=min(80,int(nRow/2),int(nCol/2))
        self.minSectionSize=10
        self.minSection=2
        self.nPC=None
        self.minVar=0.009

        self.ifFlatten=False #if flatten: each colored section is a 1d vector input; otherwise: each monochrome row in a section is a separate 1d input

        self.calParam()
        self.calParam2()
        self.find_sections()
        self.getCompressedX()
        

    #########################################################################################################
    def find_sections(self):
        rowCorr=[None]*(self.nRow-1)
        for r in range(self.nRow-1):
            rowCorr=np.corrcoef(self.image_stack[:,r,:,:].flatten(),self.image_stack[:,r+1,:,:].flatten())

        rowCorr=signal.medfilt(rowCorr,5)
        self.rowSec=self.find_locMin(rowCorr)

        colCorr=[None]*(self.nCol-1)
        for r in range(self.nCol-1):
            colCorr=np.corrcoef(self.image_stack[:,:,r,:].flatten(),self.image_stack[:,:,r+1,:].flatten())
        colCorr=signal.medfilt(colCorr,5)
        self.colSec=self.find_locMin(colCorr)
        return (self.rowSec,self.colSec)

    def find_locMin(self, arr):
        print(arr)
        res=[]
        for i in range(1,len(arr)-1):
            if arr[i]<arr[i-1] and arr[i]<arr[i+1]:
                # print(1)
                res.append(i)
        return res
    #########################################################################################################


    # rowSec,colSec=find_sections()
    # nRowSec=len(rowSec)
    # nColSec=len(colSec)

    def calParam(self):
        self.nRowSec=int(self.nRow/self.sectionSize)
        self.rowSec=np.arange(self.nRowSec+1)*self.sectionSize
        self.rowSec[-1]=max(self.rowSec[-1],self.nRow)

        self.nColSec=int(self.nCol/self.sectionSize)
        self.colSec=np.arange(self.nColSec+1)*self.sectionSize
        self.colSec[-1]=max(self.colSec[-1],self.nCol)
    # print(colSec)

        self.nSec=self.nColSec*self.nRowSec

        self.param['nRowSec']=self.nRowSec
        self.param['rowSec']=self.rowSec
        self.param['nColSec']=self.nColSec
        self.param['colSec']=self.colSec



    #preprocessing input image for PCA
    #PCA for RGB images: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.695.6281&rep=rep1&type=pdf

    def procInput_flatten(self):
        segmentedX=[None]*(self.nSec)
        for ri in range(self.nRowSec):
            for ci in range(self.nColSec):
                rowSecStart=self.rowSec[ri]
                rowSecEnd=self.rowSec[ri+1]
                colSecStart=self.colSec[ci]
                colSecEnd=self.colSec[ci+1]

                nPixel=(rowSecEnd-rowSecStart)*(colSecEnd-colSecStart)
                seci=np.zeros((self.nImages,nPixel*self.nColors))
                for img in range(self.nImages):
                    r=self.image_stack[img,rowSecStart:rowSecEnd,colSecStart:colSecEnd,0].flatten()
                    g=self.image_stack[img,rowSecStart:rowSecEnd,colSecStart:colSecEnd,1].flatten()
                    b=self.image_stack[img,rowSecStart:rowSecEnd,colSecStart:colSecEnd,2].flatten()

                    seci[img,:]=np.concatenate((r,g,b))
                # print(seci.shape)
                segmentedX[ri*self.nColSec+ci]=seci
                # print(ri*nColSec+ci)
        return segmentedX

    def procInput_noFlatten(self):
        segmentedX = [None] * (self.nSec)
        for ri in range(self.nRowSec):
            for ci in range(self.nColSec):
                rowSecStart = self.rowSec[ri]
                rowSecEnd = self.rowSec[ri + 1]
                colSecStart = self.colSec[ci]
                colSecEnd = self.colSec[ci + 1]

                self.nRow = (rowSecEnd - rowSecStart) * self.nColors
                self.nCol = (colSecEnd - colSecStart)
                seci = np.zeros((self.nImages*self.nRow, self.nCol))
                for img in range(self.nImages):
                    r = self.image_stack[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 0]
                    g = self.image_stack[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 1]
                    b = self.image_stack[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 2]

                    seci[img*self.nRow:(img+1)*self.nRow, :] = np.concatenate((r, g, b),axis=0)
                # print(seci.shape)
                segmentedX[ri * self.nColSec + ci] = seci
                # print(ri*nColSec+ci)
        return segmentedX

    ## PCA should be performed separately on each section in segmentedX
    def calParam2(self):
        self.segmentedX=None
        if self.ifFlatten:
            self.segmentedX=self.procInput_flatten()
        else:
            self.segmentedX=self.procInput_noFlatten()


    #pca and reconstruction

    def pca(self, x):
        """
        :param x: input n*d
        :return:(V,Y,Mx,w); V - eigenvector matrix (eigenvectors in columns); Y - coordinates after PCA; Mx - row mean of X, w - vector of eigenvalues
        """
        Mx=np.mean(x,axis=1)
        x_centered=np.copy(x)
        x_centered=x_centered-np.tile(np.reshape(Mx,(Mx.size,1)),x.shape[1])

        cov=np.transpose(x_centered)@x_centered
        # print(cov.shape)

        w,V=np.linalg.eig(cov)
        orderW=np.argsort(w*(-1))    #idx that sorts w in decending order
        w=w[orderW]
        V=V[:,orderW]

        # Vt=np.transpose(V)
        Y=x_centered @ V

        return (V,Y,Mx,w)


    #############################################################################################

    def pca_reconstruction(self, V,Y,Mx):
        x_centered=Y@np.transpose(V)
        x=x_centered+np.tile(np.reshape(Mx,(Mx.size,1)),x_centered.shape[1])
        return x

    def reassembleX(self, x_segmented):
        x=np.zeros((self.nImages,self.nRow,self.nCol,self.nColors))
        for ri in range(self.nRowSec):
            for ci in range(self.nColSec):
                rowSecStart = self.rowSec[ri]
                rowSecEnd = self.rowSec[ri + 1]
                colSecStart = self.colSec[ci]
                colSecEnd = self.colSec[ci + 1]

                self.nRow = (rowSecEnd - rowSecStart) * self.nColors
                nrowPerColor=(rowSecEnd - rowSecStart)
                self.nCol = (colSecEnd - colSecStart)
                seci = x_segmented[ri * self.nColSec + ci]
                for img in range(self.nImages):
                    img_seci=seci[img*self.nRow:(img+1)*self.nRow, :]

                    x[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 0]=img_seci[:nrowPerColor,:]
                    x[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 1]=img_seci[nrowPerColor:nrowPerColor*2,:]
                    x[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 2]=img_seci[nrowPerColor*2:nrowPerColor*3,:]

        return x


    # pca compression

    def pca_compression(self, x):
        V, Y, Mx, w = self.pca(x)
        w_varCap=w/np.sum(w)
        if self.minVar!=None:
            idx=np.greater(w_varCap,self.minVar)
            V=V[:,idx]
            Y=Y[:,idx]
        if self.nPC!=None:
            V=V[:,:self.nPC]
            Y=Y[:,:self.nPC]
        return (V,Y,Mx,w)


    ##tests
    # from sklearn import metrics
    ## test 1, reconstruction with all pcs
    # V,Y,Mx,w=pca(segmentedX[0])
    # reconstructedX=pca_reconstruction(V,Y,Mx)
    # print('mse:'+str(metrics.mean_squared_error(segmentedX[0],reconstructedX)))

    # ## test 2, reconstruction with 4% compression
    # print('4% compression:')
    # for s in range(len(segmentedX)):
    #     V,Y,Mx,w=pca_compression(segmentedX[s],int(segmentedX[s].shape[0]/25))
    #     nPixels=segmentedX[s].size
    #     reconstructedX=pca_reconstruction(V,Y,Mx)
    #     print('mse per pixel:'+str(metrics.mean_squared_error(segmentedX[s],reconstructedX)/nPixels))
    #
    # ## test 3, reconstruction with 1% compression
    # print('1% compression:')
    # for s in range(len(segmentedX)):
    #     V,Y,Mx,w=pca_compression(segmentedX[s],int(segmentedX[s].shape[0]/100))
    #     nPixels=segmentedX[s].size
    #     reconstructedX=pca_reconstruction(V,Y,Mx)
    #     print('mse per pixel:'+str(metrics.mean_squared_error(segmentedX[s],reconstructedX)/nPixels))

    ## test 4, reassemble frames
    # compressedX=[None]*len(segmentedX)
    # for s in range(len(segmentedX)):
    #     V,Y,Mx,w=pca_compression(segmentedX[s])
    #     nPixels=segmentedX[s].size
    #     reconstructedX=pca_reconstruction(V,Y,Mx)
    #     compressedX[s]=reconstructedX
    #     # print('mse per pixel:'+str(metrics.mean_squared_error(segmentedX[s],reconstructedX)/nPixels))
    # reassembledImg=reassembleX(compressedX)
    # assert reassembledImg.shape==frames.image_stack.shape
    # print('mse per pixel:'+str(metrics.mean_squared_error(np.reshape(frames.image_stack,frames.image_stack.size),np.reshape(reassembledImg,reassembledImg.size))/frames.image_stack.size))
    # print('max mse:'+str(np.max(metrics.mean_squared_error(np.reshape(frames.image_stack,frames.image_stack.size),np.reshape(reassembledImg,reassembledImg.size),multioutput='raw_values'))))
    # for img in range(nImages):
    #     plt.imshow(reassembledImg[img]/np.max(reassembledImg[img]))
    #     # plt.imshow(frames.image_stack[img])
    #     plt.show()

    ## test 5, reassemble frames with 4% compression
    # compressedX=[None]*len(segmentedX)
    # for s in range(len(segmentedX)):
    #     V,Y,Mx,w=pca_compression(segmentedX[s],int(segmentedX[s].shape[0]/100))
    #     nPixels=segmentedX[s].size
    #     reconstructedX=pca_reconstruction(V,Y,Mx)
    #     compressedX[s]=reconstructedX
    #     # print('mse per pixel:'+str(metrics.mean_squared_error(segmentedX[s],reconstructedX)/nPixels))
    # reassembledImg=reassembleX(compressedX)
    # assert reassembledImg.shape==frames.image_stack.shape
    # print('mse per pixel:'+str(metrics.mean_squared_error(np.reshape(frames.image_stack,frames.image_stack.size),np.reshape(reassembledImg,reassembledImg.size))/frames.image_stack.size))
    # print('max mse:'+str(np.max(metrics.mean_squared_error(np.reshape(frames.image_stack,frames.image_stack.size),np.reshape(reassembledImg,reassembledImg.size),multioutput='raw_values'))))
    # for img in range(nImages):
    #     plt.imshow(reassembledImg[img]/np.max(reassembledImg[img]))
    #     # plt.imshow(frames.image_stack[img])
    #     plt.show()

    ##sending compressed arrays
    def getCompressedX(self):
        self.calParam2()
        compressedX=None
        self.param['lenSegmentedX']=len(self.segmentedX)
        for s in range(len(self.segmentedX)):
            # V,Y,Mx,w=pca_compression(segmentedX[s],int(segmentedX[s].shape[0]*compressionRatio))
            V,Y,Mx,w=self.pca_compression(self.segmentedX[s])
            self.param['shapeV'+str(s)]=V.shape
            self.param['shapeY'+str(s)]=Y.shape
            self.param['shapeMx'+str(s)]=Mx.shape
            ci=np.concatenate((V.flatten(),Y.flatten(),Mx.flatten()))
            if s==0:
                compressedX=ci
            else:
                compressedX=np.concatenate((compressedX,ci))
        self.compressedX = compressedX
        # print(compressedX.size)

    def pca_reconstruct(self, compressedX, param):

        lenSegmentedX=param['lenSegmentedX']
        nImages=param['nImages']
        nRow=param['nRow']
        nCol=param['nCol']
        nColors=param['nColors']
        nRowSec=param['nRowSec']
        rowSec=param['rowSec']
        nColSec=param['nColSec']
        colSec=param['colSec']

        pca = PCA()
        pca.init_rec(nImages,nRow,nCol,nColors)

        reconstructX=[None]*lenSegmentedX
        for s in range(lenSegmentedX):
            Vs_shape=param['shapeV' + str(s)]
            Vs_size=Vs_shape[0]*Vs_shape[1]
            Ys_shape=param['shapeY' + str(s)]
            Ys_size=Ys_shape[0]*Ys_shape[1]
            Mx_shape=param['shapeMx' + str(s)]
            Mx_size=Mx_shape[0]

            V=np.reshape(compressedX[:Vs_size],Vs_shape)
            compressedX=compressedX[Vs_size:]

            Y=np.reshape(compressedX[:Ys_size],Ys_shape)
            compressedX=compressedX[Ys_size:]

            Mx=np.reshape(compressedX[:Mx_size],Mx_shape)
            compressedX=compressedX[Mx_size:]


            reconstructedXs=pca.pca_reconstruction(V,Y,Mx)
            reconstructX[s]=reconstructedXs
            # print('mse per pixel:'+str(metrics.mean_squared_error(segmentedX[s],reconstructedX)/nPixels))
        reassembledImg=pca.reassembleX(reconstructX)

        for img in range(nImages):
            reassembledImg[img]=reassembledImg[img]/np.max(reassembledImg[img])
            # plt.imshow(reassembledImg[img])
            # plt.imshow(frames.image_stack[img])
            # plt.savefig("frame"+str(img)+'.png')
            # plt.show()


        return reassembledImg

