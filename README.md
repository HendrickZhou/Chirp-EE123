# Chirp-EE123
final project for EE123 spring 2019

## Project Structure
This repository shoule be able be cloned into and run directly on Rasberry Pi

Before we start integrate our code, we can develop Compression and Transmission part sperately under `/src/compression` & `/src/transmission`

Our final code should run on `/src/main.ipynb`(or `main.py`, but I think notebook looks much better)

`/src/utils.py` contains utilities function for the purpose of convenience.

## Concerns about cross-platform
Basically compression and transmission part is likely to run on different platform, some inconvenience might be cause by this.

I'm not sure if cloning this whole repository into Pi would cause shortage on memory while we're developing on it, so for transmission it's probably not a bad idea to develop on your own code first.

Anyway this repository is created for the purpose of better collaboration.