#!/bin/bash

# Script to download pre-trained weights for TIF-Contrast backbones
mkdir -p weights
cd weights

echo "Downloading SciBERT weights..."
# Example URL for SciBERT
# wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar.gz

echo "Downloading CLIP (ViT-B/32) weights..."
# CLIP weights are usually handled by the open_clip library automatically, 
# but you can pre-cache them here.

echo "Downloading MathBERT weights..."
# git clone https://github.com/tbs17/MathBERT.git

echo "All weight directories initialized."