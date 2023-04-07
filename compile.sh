#!/bin/bash

echo "----- COMPILING -----"

cd src

/usr/local/cuda/bin/nvcc equiToCube.cu `pkg-config opencv4 --cflags --libs` equiToCube.cpp utils.cpp -o ../build/equiToCube
/usr/local/cuda/bin/nvcc cubeToEqui.cu `pkg-config opencv4 --cflags --libs` cubeToEqui.cpp utils.cpp -o ../build/cubeToEqui

rm -f ./*.o