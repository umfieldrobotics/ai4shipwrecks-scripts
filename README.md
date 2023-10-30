# ai4shipwrecks-scripts

This repository containts scripts to aid in the usage of the open source AI4Shipwrecks dataset. The paper website can be found at https://umfieldrobotics.github.io/ai4shipwrecks/.

## `generate_square_images.py`

This script takes in a folder of full-sized sonar images and labels, and then produces square overlapping cropped images (and corresponding labels) `STRIDE` pixels apart.

## `dataset.py`

This script contains the PyTorch Dataset that we used in training the baselines.
