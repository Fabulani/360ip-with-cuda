# 360° Image Processing with CUDA and OpenCV

This project was developed in the 3-day sprint for my _Real-time image processing with CPUs and GPUs_ class in University of Jean-Monnet Saint-Étienne (UJM) for my Master program IMLEX. The goal of this project is to experiment with different formats of 360° images and videos, and apply image processing techniques using openCV and CUDA.

![Equirectangular Image](./data/image-360.jpg)
![Cubemap Image](./data/image-360-cubemap.jpg)

# Goals

The following goals were set:
- [x] implement the following image transformations with OpenCV and CUDA:
  - [x] Equirectangular => Cube map
  - [x] Cube map => Equirectangular
- [ ] apply image processing methods (e.g., gaussian blur) to an equirectangular image and display it with `image-360.html`
- [ ] apply image processing methods to each face of a cube map and transform the result in its equirectangular representation, then display it with `image-360.html`
- [ ] apply the cube based method on normal and stereo videos

# Prerequisites

- C++ compiler
- OpenCV
- CUDA

# Data

Data used is a self-capture of 360° equirectangular images, which were used to generate the cube map images. A 360° video was also used and can be obtained by downloading from [here](https://drive.google.com/drive/folders/1VfOEQuCta-riJXZV4CCk2Hx9R8peUIZt?usp=sharing) or by using `wget`:

```sh
wget https://drive.google.com/file/d/11EXntJ0xs5AxysyZY65hkW75XnG7FZTw/view?usp=share_link
```

# Folder structure

The project is organized as follows:
- `build`: contains generated executables
- `data`: contains the data (i.e., images and videos). Program output goes to `data/out`.
- `docs`: contains documentation (mostly for this readme)
- `src`: contains source code (`.cpp`, `.cu`, `.html`, etc)

# Compiling and executing

To compile, simply run `sh compile.sh`.

To execute, pass in as an argument the path to the image file:
```sh
# Equirectangular to cubemap conversion:
./build/equiToCube ./data/image-360.jpg

# Cubemap to equirectangular conversion:
./build/cubeToEqui ./data/image-360-cubemap.jpg
```
