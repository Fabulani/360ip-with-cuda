#!/bin/bash

echo "----- COMPILING -----"

cd src

/usr/local/cuda/bin/nvcc cuda.cu `pkg-config opencv4 --cflags --libs` cuda.cpp -o ../build/cuda

rm -f ./*.o