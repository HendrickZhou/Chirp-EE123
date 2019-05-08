<<<<<<< HEAD
import compression.loadData as loadData
=======
import loadData
import os
>>>>>>> 9f0bce8475ef877c47ea5d5f312fd922282a0299
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

frames = loadData.LoadData()
frames.loadVideoPixelData('../../asset/Andy_Video.png')

print("shape of video data %s" % (frames.image_stack.shape,))
nImages,nRow,nCol,nColors=frames.image_stack.shape      #assumes 3 color channel when preprocess

param={}
param['nImages']=nImages
param['nRow']=nRow
param['nCol']=nCol
param['nColors']=nColors

#tunable parameters
compressionRatio=0.005
sectionSize=min(80,int(nRow/2),int(nCol/2))
minSectionSize=10
minSection=2
nPC=None
minVar=0.009

def find_sections():
    rowCorr=[None]*(nRow-1)
    for r in range(nRow-1):
        rowCorr=np.corrcoef(frames.image_stack[:,r,:,:].flatten(),frames.image_stack[:,r+1,:,:].flatten())

    rowCorr=signal.medfilt(rowCorr,5)
    rowSec=find_locMin(rowCorr)

    colCorr=[None]*(nCol-1)
    for r in range(nCol-1):
        colCorr=np.corrcoef(frames.image_stack[:,:,r,:].flatten(),frames.image_stack[:,:,r+1,:].flatten())
    colCorr=signal.medfilt(colCorr,5)
    colSec=find_locMin(colCorr)
    return (rowSec,colSec)

def find_locMin(arr):
    print(arr)
    res=[]
    for i in range(1,len(arr)-1):
        if arr[i]<arr[i-1] and arr[i]<arr[i+1]:
            # print(1)
            res.append(i)
    return res


ifFlatten=False #if flatten: each colored section is a 1d vector input; otherwise: each monochrome row in a section is a separate 1d input

# rowSec,colSec=find_sections()
# nRowSec=len(rowSec)
# nColSec=len(colSec)


nRowSec=int(nRow/sectionSize)
rowSec=np.arange(nRowSec+1)*sectionSize
rowSec[-1]=max(rowSec[-1],nRow)
print(rowSec)

nColSec=int(nCol/sectionSize)
colSec=np.arange(nColSec+1)*sectionSize
colSec[-1]=max(colSec[-1],nCol)
# print(colSec)

nSec=nColSec*nRowSec

param['nRowSec']=nRowSec
param['rowSec']=rowSec
param['nColSec']=nColSec
param['colSec']=colSec



#preprocessing input image for PCA
#PCA for RGB images: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.695.6281&rep=rep1&type=pdf

def procInput_flatten():
    segmentedX=[None]*(nSec)
    for ri in range(nRowSec):
        for ci in range(nColSec):
            rowSecStart=rowSec[ri]
            rowSecEnd=rowSec[ri+1]
            colSecStart=colSec[ci]
            colSecEnd=colSec[ci+1]

            nPixel=(rowSecEnd-rowSecStart)*(colSecEnd-colSecStart)
            seci=np.zeros((nImages,nPixel*nColors))
            for img in range(nImages):
                r=frames.image_stack[img,rowSecStart:rowSecEnd,colSecStart:colSecEnd,0].flatten()
                g=frames.image_stack[img,rowSecStart:rowSecEnd,colSecStart:colSecEnd,1].flatten()
                b=frames.image_stack[img,rowSecStart:rowSecEnd,colSecStart:colSecEnd,2].flatten()

                seci[img,:]=np.concatenate((r,g,b))
            # print(seci.shape)
            segmentedX[ri*nColSec+ci]=seci
            # print(ri*nColSec+ci)
    return segmentedX

def procInput_noFlatten():
    segmentedX = [None] * (nSec)
    for ri in range(nRowSec):
        for ci in range(nColSec):
            rowSecStart = rowSec[ri]
            rowSecEnd = rowSec[ri + 1]
            colSecStart = colSec[ci]
            colSecEnd = colSec[ci + 1]

            nrow = (rowSecEnd - rowSecStart) * nColors
            ncol = (colSecEnd - colSecStart)
            seci = np.zeros((nImages*nrow, ncol))
            for img in range(nImages):
                r = frames.image_stack[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 0]
                g = frames.image_stack[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 1]
                b = frames.image_stack[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 2]

                seci[img*nrow:(img+1)*nrow, :] = np.concatenate((r, g, b),axis=0)
            # print(seci.shape)
            segmentedX[ri * nColSec + ci] = seci
            # print(ri*nColSec+ci)
    return segmentedX

## PCA should be performed separately on each section in segmentedX
segmentedX=None
if ifFlatten:
    segmentedX=procInput_flatten()
else:
    segmentedX=procInput_noFlatten()


#pca and reconstruction

def pca(x):
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

def pca_reconstruction(V,Y,Mx):
    x_centered=Y@np.transpose(V)
    x=x_centered+np.tile(np.reshape(Mx,(Mx.size,1)),x_centered.shape[1])
    return x

def reassembleX(x_segmented):
    x=np.zeros((nImages,nRow,nCol,nColors))
    for ri in range(nRowSec):
        for ci in range(nColSec):
            rowSecStart = rowSec[ri]
            rowSecEnd = rowSec[ri + 1]
            colSecStart = colSec[ci]
            colSecEnd = colSec[ci + 1]

            nrow = (rowSecEnd - rowSecStart) * nColors
            nrowPerColor=(rowSecEnd - rowSecStart)
            ncol = (colSecEnd - colSecStart)
            seci = x_segmented[ri * nColSec + ci]
            for img in range(nImages):
                img_seci=seci[img*nrow:(img+1)*nrow, :]

                x[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 0]=img_seci[:nrowPerColor,:]
                x[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 1]=img_seci[nrowPerColor:nrowPerColor*2,:]
                x[img, rowSecStart:rowSecEnd, colSecStart:colSecEnd, 2]=img_seci[nrowPerColor*2:nrowPerColor*3,:]

    return x


# pca compression

def pca_compression(x,nPC=None,minVar=None):
    V, Y, Mx, w = pca(x)
    w_varCap=w/np.sum(w)
    if minVar!=None:
        idx=np.greater(w_varCap,minVar)
        V=V[:,idx]
        Y=Y[:,idx]
    if nPC!=None:
        V=V[:,:nPC]
        Y=Y[:,:nPC]
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
<<<<<<< HEAD
#     V,Y,Mx,w=pca_compression(segmentedX[s],int(segmentedX[s].shape[0]/25))
=======
#     V,Y,Mx,w=pca_compression(segmentedX[s],int(segmentedX[s].shape[0]/100))
>>>>>>> 9f0bce8475ef877c47ea5d5f312fd922282a0299
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

<<<<<<< HEAD
# sending compressed arrays
compressedX=None
for s in range(len(segmentedX)):
    V,Y,Mx,w=pca_compression(segmentedX[s],int(segmentedX[s].shape[0]/25))
#     print('shape of V:'+str(V.shape))
#     print('shape of Y:'+str(Y.shape))
#     print('shape of Mx:'+str(Mx.shape))
    compressedXi=np.concatenate((V.flatten(),Y.flatten(),Mx.flatten()))
    print(compressedXi.shape)
    if s==0:
        compressedX=compressedXi
    else:
        compressedX=np.concatenate((compressedX,compressedXi))
=======
##sending compressed arrays
compressedX=None
param['lenSegmentedX']=len(segmentedX)
for s in range(len(segmentedX)):
    # V,Y,Mx,w=pca_compression(segmentedX[s],int(segmentedX[s].shape[0]*compressionRatio))
    V,Y,Mx,w=pca_compression(segmentedX[s],minVar=minVar)
    param['shapeV'+str(s)]=V.shape
    param['shapeY'+str(s)]=Y.shape
    param['shapeMx'+str(s)]=Mx.shape
    ci=np.concatenate((V.flatten(),Y.flatten(),Mx.flatten()))
    if s==0:
        compressedX=ci
    else:
        compressedX=np.concatenate((compressedX,ci))

print(compressedX.size)


>>>>>>> 9f0bce8475ef877c47ea5d5f312fd922282a0299
