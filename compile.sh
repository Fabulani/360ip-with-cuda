#!/bin/bash

echo "----- COMPILING -----"

cd src

/usr/local/cuda/bin/nvcc equiToCube.cu cubeToEqui.cu `pkg-config opencv4 --cflags --libs` main.cpp utils.cpp -o ../build/main

rm -f ./*.o