#include <stdio.h>
#include <stdlib.h>

#include <cfloat>
#include <cmath>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/opencv.hpp>

/* ------------------------------------------------------------------------------------------------------------------------
#
# Code adapted from Paul Reed.
# Github: https://github.com/PaulMakesStuff/Cubemaps-Equirectangular-DualFishEye/blob/master/CubemapFromEqui.py
#
# ------------------------------------------------------------------------------------------------------------------------ */

__device__ float getTheta(float x, float y) {
    float rtn = 0;
    if (y < 0) {
        rtn = atan2f(y, x) * -1;
    } else {
        rtn = M_PI + (M_PI - atan2f(y, x));
    }
    return rtn;
}

__global__ void equiToCube(cv::cuda::PtrStep<uchar3> input, cv::cuda::PtrStep<uchar3> output, const int in_rows, const int in_cols, const int out_rows, const int out_cols, const float sqr) {
    /*----------------------------------------------------------------------------------------------------------
    #
    # take an equirectangular image and convert it to a cube map image of the following format
    #
    #	+----+----+----+
    #	| Y- | X+ | Y+ |
    #	+----+----+----+
    #	| X- | Z- | Z+ |
    #	+----+----+----+
    #
    # which when unfolded takes the following format
    #
    #	+----+
    #	| Z+ |
    #	+----+----+----+----+
    #	| X+ | Y+ | X- | Y- |
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

    int tx = 0;
    int ty = 0;
    float x = 0;
    float y = 0;
    float z = 0;

    // Boxes for the equirectangular
    if (row < sqr + 1) {
        // TOP HALF
        if (col < sqr + 1) {
            // TOP LEFT [Y+]
            tx = col;
            ty = row;
            x = tx - 0.5 * sqr;
            y = 0.5 * sqr;
            z = ty - 0.5 * sqr;
        } else if (col < 2 * sqr + 1) {
            // TOP MIDDLE [X+]
            tx = col - sqr;
            ty = row;
            x = 0.5 * sqr;
            y = (tx - 0.5 * sqr) * (-1);
            z = ty - 0.5 * sqr;
        } else {
            // TOP RIGHT [Y-]
            tx = col - sqr * 2;
            ty = row;
            x = (tx - 0.5 * sqr) * (-1);
            y = -0.5 * sqr;
            z = ty - 0.5 * sqr;
        }
    } else {
        // BOTTOM HALF
        if (col < sqr + 1) {
            // BOTTOM LEFT [X-]
            tx = col;
            ty = row - sqr;
            x = int(-0.5 * sqr);
            y = int(tx - 0.5 * sqr);
            z = int(ty - 0.5 * sqr);
        } else if (col < 2 * sqr + 1) {
            // BOTTOM MIDDLE [Z-]
            tx = col - sqr;
            ty = row - sqr;
            x = (ty - 0.5 * sqr) * (-1);
            y = (tx - 0.5 * sqr) * (-1);
            z = 0.5 * sqr;
        } else {
            // BOTTOM RIGHT [Z+]
            tx = col - sqr * 2;
            ty = row - sqr;
            x = ty - 0.5 * sqr;
            y = (tx - 0.5 * sqr) * (-1);
            z = -0.5 * sqr;
        }
    }

    // Now for the polar coordinates
    float rho = sqrt(x * x + y * y + z * z);
    float norm_theta = getTheta(x, y) / (2 * M_PI);  // Normalized theta
    float norm_phi = (M_PI - acos(z / rho)) / M_PI;  // Normalized phi

    // For the coordinates
    int iRow = norm_phi * in_rows;
    int iCol = norm_theta * in_cols;

    // Catch possible overflows
    if (iRow >= in_rows)
        iRow -= in_rows;
    if (iCol >= in_cols)
        iCol -= in_cols;

    // Apply to output image
    output(row, col) = input(iRow, iCol);
}

int divUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat &src, cv::cuda::GpuMat &output, float sqr) {
    const dim3 block(32, 8);
    const dim3 grid(divUp(output.cols, block.x), divUp(output.rows, block.y));

    int in_cols = src.size().width;
    int in_rows = src.size().height;
    int out_cols = output.size().width;
    int out_rows = output.size().height;

    equiToCube<<<grid, block>>>(src, output, in_rows, in_cols, out_rows, out_cols, sqr);
}
