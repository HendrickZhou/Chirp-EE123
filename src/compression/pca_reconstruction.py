import pca
import numpy as np
import matplotlib.pyplot as plt
# param= pca.param
# compressedX=pca.compressedX
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