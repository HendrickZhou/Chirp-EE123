import sys
import os
script_path = os.path.abspath('')
module_path = script_path[:script_path.rfind('src')]+ 'src' + '/'
asset_path = script_path[:script_path.rfind('src')]+ 'asset' + '/'
sys.path.append(module_path)
import numpy as np
import matplotlib.pyplot as plt
import pca

from utils.Helper_functions import imageStack_load

image_stack = imageStack_load(asset_path+"dog.png")

pca_example=pca.PCA(image_stack)
pca_example.procInput_noFlatten()
compressedX=pca_example.getArraysToTransmit()
encodingPCA=pca_example.encode_PCA(compressedX)
# print(np.max(compressedX))
# print(np.min(compressedX))
# print(compressedX[0])
# print(np.binary_repr(compressedX[0]))

decodedX,param=pca.decode_PCA(encodingPCA)
reconstructed=pca.pca_reconstruct(decodedX,param)

print(compressedX[:100])
# print(param)

print(reconstructed.shape)
