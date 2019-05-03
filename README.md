# Chirp-EE123
final project for EE123 spring 2019

## Project Structure
This repository shoule be able be cloned into and run directly on Rasberry Pi

Before we start integrate our code, we can develop Compression and Transmission part sperately under `/src/compression` & `/src/transmission`

Our final code should run on `/src/main.ipynb`(or `main.py`, but I think notebook looks much better)

`/src/utils/` contains utilities function for the purpose of convenience.

our test videos/images will be put under `/asset`

## Notice!
Before we find better solution for the python module import problem, remeber to change the `sys.path.append("path")` to your own project directory!

I'm not pretty sure if this would work on Rasberry Pi.

## Concerns about cross-platform
Basically compression and transmission part is likely to run on different platform, some inconvenience might be cause by this.

I'm not sure if cloning this whole repository into Pi would cause shortage on memory while we're developing on it, so for transmission it's probably not a bad idea to develop on your own code first.

Anyway this repository is created for the purpose of better collaboration.





## APIs
-   Import the helper functions:

`from utils.Helper_functions import *`

-   Load PNG stack as 4D numpy array (frame#, Y, X, RGB)

`imageStack_load(filename = 'path/FILENAME.png')`

-   Save Tiff images into a PNG stack with 10 frame/sec frame rate

`GIF_save(path = 'path/', framerate = 10)`

-   Display Tiff stack as video with display window size of 400 and frame rate of 10 frame/sec

`Tiff_play(path= 'path/*.tiff', display_size=400, frame_rate=10)`

-    Compute peak signal-to-noise ratio (PSNR):

`psnr(reference_image_stack, measured_image_stack, maxVal=255)`
