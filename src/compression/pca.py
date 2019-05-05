import loadData
import numpy as np

frames = loadData.LoadData()
frames.loadVideoPixelData('C:\\UCBERKELEY\\sp19\\ee123\\project\\Chirp-EE123\\asset\\Andy_Video.png')

print("shape of video data %s" % (frames.image_stack.shape,))
nImages,nRow,nCol,nColors=frames.image_stack.shape      #assumes 3 color channel when preprocess

#tunable parameters
sectionSize=80
ifFlatten=False #if flatten: each colored section is a 1d vector input; otherwise: each monochrome row in a section is a separate 1d input

#TODO: implement sectioning by similarity: http://ijcscn.com/Documents/Volumes/vol2issue1/ijcscn2012020118.pdf
nRowSec=int(nRow/sectionSize)
rowSec=np.arange(nRowSec+1)*sectionSize
rowSec[-1]=max(rowSec[-1],nRow)
print(rowSec)

nColSec=int(nCol/sectionSize)
colSec=np.arange(nColSec+1)*sectionSize
colSec[-1]=max(colSec[-1],nCol)
print(colSec)

nSec=nColSec*nRowSec

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

# pca compression

def pca_compression(x,nPC=None):
    V, Y, Mx, w = pca(x)
    if nPC!=None:
        V=V[:,:nPC]
        Y=Y[:,:nPC]
    return (V,Y,Mx,w)


##tests
from sklearn import metrics
## test 1, reconstruction with all pcs - passed
# V,Y,Mx,w=pca(segmentedX[0])
# reconstructedX=pca_reconstruction(V,Y,Mx)
# print('mse:'+str(metrics.mean_squared_error(segmentedX[0],reconstructedX)))

## test 2, reconstruction with 4% compression
print('4% compression:')
for s in range(len(segmentedX)):
    V,Y,Mx,w=pca_compression(segmentedX[s],int(segmentedX[s].shape[0]/25))
    nPixels=segmentedX[s].size
    reconstructedX=pca_reconstruction(V,Y,Mx)
    print('mse per pixel:'+str(metrics.mean_squared_error(segmentedX[s],reconstructedX)/nPixels))

## test 3, reconstruction with 1% compression
print('1% compression:')
for s in range(len(segmentedX)):
    V,Y,Mx,w=pca_compression(segmentedX[s],int(segmentedX[s].shape[0]/100))
    nPixels=segmentedX[s].size
    reconstructedX=pca_reconstruction(V,Y,Mx)
    print('mse per pixel:'+str(metrics.mean_squared_error(segmentedX[s],reconstructedX)/nPixels))