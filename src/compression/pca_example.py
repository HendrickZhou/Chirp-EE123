import numpy as np
import matplotlib.pyplot as plt
import loadData
import pca_reconstruction

frames = loadData.LoadData()
frames.loadVideoPixelData('C:\\UCBERKELEY\\sp19\\ee123\\project\\Chirp-EE123\\asset\\Andy_Video.png')

import pca
compressedX,param=pca.compressedX,pca.param
reconstructedFrames=pca_reconstruction.pca_reconstruct(compressedX,param)