import numpy as np
import matplotlib.pyplot as plt
import loadData
import pca_reconstruction
import pca_clean

frames = loadData.LoadData()
frames.loadVideoPixelData('C:\\UCBERKELEY\\sp19\\ee123\\project\\Chirp-EE123\\asset\\Andy_Video.png')

pca_example=pca_clean.PCA(frames.image_stack)
pca_example.procInput_noFlatten()
compressedX,param=pca_example.getArraysToTransmit()

reconstructed=pca_clean.pca_reconstruct(compressedX,param)