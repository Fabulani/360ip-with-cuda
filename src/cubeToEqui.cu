#include <stdio.h>
#include <stdlib.h>

#include <cfloat>
#include <cmath>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/opencv.hpp>

#include "utils.h"

/* ------------------------------------------------------------------------------------------------------------------------
#
# Code adapted from Paul Reed.
# Github: https://github.com/PaulMakesStuff/Cubemaps-Equirectangular-DualFishEye/blob/master/createEquiFromSquareFiles.py
#
# ------------------------------------------------------------------------------------------------------------------------ */

enum class CubeFace {
    Xp,  // X+
    Xm,  // X-
    Yp,  // Y+
    Ym,  // Y-
    Zp,  // Z+
    Zm   // Z-
};

enum class Axis {
    X,
    Y,
    Z
};

struct RayCollision {
    // The 3D coordinates of the point where the ray hits the cube
    float x;
    float y;
    float z;
    // Which face of the cube is hit by the ray
    CubeFace face;
};

struct CubeFaceCoord2D {
    float x;
    float y;
    CubeFace face;
};

__device__ float2
coord3DTo2D(RayCollision ray_collision) {
    /*
    Convert from 3D to 2D coordinates according to the ray collision face.

    Return the 2D coordinates from the cube face space.
    */

    float2 coord2D;

    // Different faces have different axis directions. Apply transform for each case.
    switch (ray_collision.face) {
        case CubeFace::Xp:
            coord2D.x = ray_collision.y + 0.5;
            coord2D.y = ray_collision.z + 0.5;
            break;
        case CubeFace::Yp:
            coord2D.x = (ray_collision.x * -1) + 0.5;
            coord2D.y = ray_collision.z + 0.5;
            break;
        case CubeFace::Xm:
            coord2D.x = (ray_collision.y * -1) + 0.5;
            coord2D.y = ray_collision.z + 0.5;
            break;
        case CubeFace::Ym:
            coord2D.x = ray_collision.x + 0.5;
            coord2D.y = ray_collision.z + 0.5;
            break;
        case CubeFace::Zp:
            coord2D.x = ray_collision.y + 0.5;
            coord2D.y = (ray_collision.x * -1) + 0.5;
            break;
        case CubeFace::Zm:
            coord2D.x = ray_collision.y + 0.5;
            coord2D.y = ray_collision.x + 0.5;
            break;
        default:
            printf("ERROR! No cubeface detected in coord3Dto2D()!\n");
    }

    // Images y axis start from top left of the image. Adjust coord2D.y for this.
    coord2D.y = 1 - coord2D.y;

    return coord2D;
}

__device__ RayCollision projectRay(const float theta, const float phi, const float sign, const Axis axis) {
    /*
    Project a ray from the center of the sphere to a face of the unit cube.

    Return the ray collision 3D coordinates and the cube face hit.
    */

    RayCollision ray_collision;
    float rho;

    switch (axis) {
        case Axis::X:
            sign == 1 ? ray_collision.face = CubeFace::Xp : ray_collision.face = CubeFace::Xm;
            ray_collision.x = sign * 0.5;
            rho = ray_collision.x / (cosf(theta) * sinf(phi));
            ray_collision.y = rho * sinf(theta) * sinf(phi);
            ray_collision.z = rho * cosf(phi);
            break;
        case Axis::Y:
            sign == 1 ? ray_collision.face = CubeFace::Yp : ray_collision.face = CubeFace::Ym;
            ray_collision.y = sign * 0.5;
            rho = ray_collision.y / (sinf(theta) * sinf(phi));
            ray_collision.x = rho * cosf(theta) * sinf(phi);
            ray_collision.z = rho * cosf(phi);
            break;
        case Axis::Z:
            sign == 1 ? ray_collision.face = CubeFace::Zp : ray_collision.face = CubeFace::Zm;
            ray_collision.z = sign * 0.5;
            rho = ray_collision.z / cosf(phi);
            ray_collision.x = rho * cosf(theta) * sinf(phi);
            ray_collision.y = rho * sinf(theta) * sinf(phi);
            break;
    }
    // printf("theta: %.2f | phi: %.2f | sign: %d | ray %.2f, %.2f, %.2f \n", theta, phi, sign, ray_collision.x, ray_collision.y, ray_collision.z);
    return ray_collision;
}

__device__ float2 cubeFaceToGlobalCoord(const CubeFaceCoord2D cube_face_coord2D, const float sqr) {
    /*
    Get the global coordinates for the given cube face coordinates.
    */
    float2 offset;

    // Get the normalized offset according to the cube face
    switch (cube_face_coord2D.face) {
        case CubeFace::Xp:
            offset.x = 1;
            offset.y = 0;
            break;
        case CubeFace::Xm:
            offset.x = 0;
            offset.y = 1;
            break;
        case CubeFace::Yp:
            offset.x = 2;
            offset.y = 0;
            break;
        case CubeFace::Ym:
            offset.x = 0;
            offset.y = 0;
            break;
        case CubeFace::Zp:
            offset.x = 2;
            offset.y = 1;
            break;
        case CubeFace::Zm:
            offset.x = 1;
            offset.y = 1;
            break;
    }

    // Multiply by the length of the square (face) to obtain the correct coord offset
    offset.x *= sqr;
    offset.y *= sqr;

    // Add the offset to the cube face coordinate, i.e. apply a transform to get the global coord
    float2 global_coord2D;
    global_coord2D.x = cube_face_coord2D.x + offset.x;
    global_coord2D.y = cube_face_coord2D.y + offset.y;
    // printf("face: %d | cubefacecoord2d: %d, %d | offset: %.2f, %.2f | globalcoord2d %.2f, %.2f \n", static_cast<int>(cube_face_coord2D.face), cube_face_coord2D.x, cube_face_coord2D.y, offset.x, offset.y, global_coord2D.x, global_coord2D.y);
    return global_coord2D;
}

__device__ CubeFaceCoord2D equiUVtoUnit2D(float theta, float phi, float sqr) {
    /*
    Convert from equirectangular image polar coordinates to cube face 2D coordinates.
    */

    // Calculate the unit vector
    float x = cosf(theta) * sinf(phi);
    float y = sinf(theta) * sinf(phi);
    float z = cosf(phi);

    // Find the maximum value in the unit vector
    float maximum = max(abs(x), max(abs(y), abs(z)));

    // Get the sign of the ray. One of the following will be 1 or -1
    float xx = x / maximum;
    float yy = y / maximum;
    float zz = z / maximum;

    // Project ray from the sphere center to the cube surface
    RayCollision ray_collision;
    if (xx == 1 || xx == -1) {
        ray_collision = projectRay(theta, phi, xx, Axis::X);
    } else if (yy == 1 || yy == -1) {
        ray_collision = projectRay(theta, phi, yy, Axis::Y);
    } else if (zz == 1 || zz == -1) {
        ray_collision = projectRay(theta, phi, zz, Axis::Z);
    } else {
        printf("ERROR! None of the signs are 1 or -1 in equiUVtoUnit2D()!");
    }

    // Convert the 3D ray collision point to the 2D coordinates in the collided cube face (normalized 0..1)
    float2 coord2D = coord3DTo2D(ray_collision);

    // Adjust coordinates by the size of the square (i.e., face length). A.k.a., denormalize the coordinates.
    coord2D.x *= sqr;
    coord2D.y *= sqr;
    CubeFaceCoord2D cubeCoord2D = {
        .x = coord2D.x,
        .y = coord2D.y,
        .face = ray_collision.face};
    return cubeCoord2D;
}

__device__ uchar3 debug_cubemap(CubeFace face) {
    /*
    Return a different colored pixel for each face. For debugging.
    */
    uchar3 pixel = make_uchar3(0, 0, 0);
    switch (face) {
        case CubeFace::Xp:
            pixel.z = 255;
            break;
        case CubeFace::Xm:
            pixel.y = 255;
            pixel.z = 255;
            break;
        case CubeFace::Yp:
            pixel.x = 255;
            break;
        case CubeFace::Ym:
            pixel.x = 255;
            pixel.y = 255;
            pixel.z = 255;
            break;
        case CubeFace::Zp:
            pixel.y = 255;
            break;
        case CubeFace::Zm:
            break;
    }
    return pixel;
}

__global__ void
cubeToEqui(cv::cuda::PtrStep<uchar3> input, cv::cuda::PtrStep<uchar3> output, const int in_rows, const int in_cols, const int out_rows, const int out_cols, const float sqr) {
    /*----------------------------------------------------------------------------------------------------------
    #
    # take a cubemap image and convert it to an equirectangular image. The cubemap has the following format:
    #
    #	+----+----+----+
    #	| Y+ | X+ | Y- |
    #	+----+----+----+
    #	| X- | Z- | Z+ |
    #	+----+----+----+
    #
    # which when unfolded takes the following format
    #
    #	+----+
    #	| Z+ |
    #	+----+----+----+----+
    #	| X+ | Y- | X- | Y+ |
    #	+----+----+----+----+
    #	| Z- |
    #	+----+
    #
    #-------------------------------------------------------------------------------------------------------*/

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= out_rows || col >= out_cols) {
        return;
    }

    // Normalized uv coordinates for the current pixel
    float u = (float)col / (out_cols - 1);
    float v = (float)row / (out_rows - 1);

    // Polar coords for the current pixel
    float theta = u * 2 * M_PI;
    float phi = v * M_PI;

    // Calculate the 3D cartesian coordinate which has been projected to a cube face
    CubeFaceCoord2D cube_face_coord2D = equiUVtoUnit2D(theta, phi, sqr);

    // Convert from cube face coordinates (local) to global coordinates
    float2 global_coord2D = cubeFaceToGlobalCoord(cube_face_coord2D, sqr);

    // Get the pixel values from the equirectangular image
    uchar3 pixel = input((int)global_coord2D.y, (int)global_coord2D.x);

    // Apply to output image
    output(row, col) = pixel;
}

void cudaCubeToEqui(cv::cuda::GpuMat& src, cv::cuda::GpuMat& output, float sqr) {
    const dim3 block(32, 8);
    const dim3 grid(divUp(output.cols, block.x), divUp(output.rows, block.y));

    int in_cols = src.size().width;
    int in_rows = src.size().height;
    int out_cols = output.size().width;
    int out_rows = output.size().height;

    cubeToEqui<<<grid, block>>>(src, output, in_rows, in_cols, out_rows, out_cols, sqr);
}
