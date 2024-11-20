## Overview

This repository contains PyTorch ECG classification model architectures of four types, namely 
* A model based on InceptionTime (with varying amounts of initial downsampling by repeated strided convolutions, either 2x, 4x or 8x).
* A 1D-CNN, reimplemented based on the description in ''Screening for cardiac contractile dysfunction using an artificial intelligence–enabled electrocardiogram'' published in Nature Medicine 2019.
* An attention based network taking spectrograms as input.
* A 2D-CNN inspired by ResNet, taking spectrograms as input, reimplemented based on description in ''Artificial Intelligence Algorithm for Screening Heart Failure with Reduced Ejection Fraction Using Electrocardiography'' published in International Jounal of Cardiology.

The models were implemented and used with python version 3.8.18 and pytorch version 1.10.2
