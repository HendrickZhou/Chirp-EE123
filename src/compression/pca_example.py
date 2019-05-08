import sys
import numpy as np
import matplotlib.pyplot as plt
import os
script_path = os.path.abspath('')
module_path = script_path[:script_path.rfind('src')]+ 'src' + '/'
asset_path = script_path[:script_path.rfind('src')]+ 'asset' + '/'
sys.path.append(module_path)
from utils.Helper_functions import *
from pca import PCA

image_stack = imageStack_load(asset_path+"simpson.png")

pca = PCA()
pca.init(image_stack)
compressedX,param=pca.compressedX,pca.param

pca_r = PCA()
reconstructedFrames=pca_r.pca_reconstruct(compressedX,param)